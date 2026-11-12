"""Resource manifests survive abrupt process exit and drive confirmed orphan cleanup."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from agent_env.config import configure, get_config
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.runner import store as run_store
from agent_env.runner.local_runner import LocalRunner
from agent_env.runner.runner import RunStatus
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task import Task
from agent_env.task import teardown as teardown_api
from agent_env.task.child_store import TASK_CHILD_INVOCATIONS_COLLECTION, ChildCleanupState, get_child_invocation_store
from agent_env.task.execution import TaskLifecycleError
from agent_env.task.resources import record_resource_context
from agent_env.task.store import get_task_instance_store
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep


TEST_BUDGET_SECONDS = 5.0
PROCESS_BUDGET_SECONDS = 15.0
STAGED_CHANGELOG_ENTRY = {
    "agent_name": "worker", "staging_url": "https://fixture.invalid/staging/private",
    "object_url": "file:///fixture/changelog",
}


class RecoveryProvider(SandboxProvider):
    ON_THIS_MACHINE = False

    def __init__(self, *, log_path: str, failure_path: str) -> None:
        self.log_path = Path(log_path)
        self.failure_path = Path(failure_path)

    async def create_sandbox(self, **parameters) -> Sandbox:
        raise AssertionError("Recovery tests record resources without provisioning compute")

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        raise AssertionError("Recovery must terminate the recorded resource by id")

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        if self.failure_path.exists():
            raise RuntimeError("termination was not confirmed")
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{sandbox_id}\n")


class RecoveryStep(TaskStep):
    type = "resource_recovery_probe"
    entity_refs = ()
    calls = Counter()

    def __init__(self, *, id: str, version: int | None = None, resource_id: str = "", action: str = "return") -> None:
        super().__init__(id=id, version=version)
        self.resource_id = resource_id
        self.action = action

    def to_dict(self) -> dict:
        return {**super().to_dict(), "resource_id": self.resource_id, "action": self.action}

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryStep:
        return cls(id=data["id"], version=data.get("version"), resource_id=data["resource_id"], action=data["action"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        self.calls[self.id] += 1
        if self.action == "lifecycle":
            raise TaskLifecycleError("resource ownership cannot be confirmed")
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name=self.id, sandbox_id=self.resource_id, sandbox_mode="container", sandbox_type="recovery-test",
        ))
        if self.action in ("changelog", "changelog_fail"):
            context.metadata["agent_changelog"] = [dict(STAGED_CHANGELOG_ENTRY)]
        await context.checkpoint_resources()
        if self.action == "crash":
            os._exit(17)
        if self.action in ("fail", "changelog_fail"):
            raise RuntimeError("step failed after recording its resource")
        return context


@pytest.fixture
def recovery_config(tmp_path, monkeypatch):
    database = tmp_path / "documents.sqlite3"
    termination_log = tmp_path / "terminated.txt"
    failure_marker = tmp_path / "fail-termination"
    config = tmp_path / "config.toml"
    config.write_text(
        '[task_steps]\nimpls = ["tst.unit.task.resource_recovery_test:RecoveryStep"]\n'
        '[sandbox.providers.recovery-test]\n'
        'impl = "tst.unit.task.resource_recovery_test:RecoveryProvider"\n'
        '[sandbox.providers.recovery-test.config]\n'
        f'log_path = "{termination_log}"\nfailure_path = "{failure_marker}"\n', encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    configure(document_store=LocalSqliteDocumentStore(path=str(database)))
    RecoveryStep.calls.clear()
    return database, termination_log, failure_marker


def nested_step(task: Task, *, fail_task_on_error: bool = True) -> SubDAGTaskStep:
    return SubDAGTaskStep(
        id="nested", version=None, task=task, child_timeout_seconds=TEST_BUDGET_SECONDS,
        cleanup_timeout_seconds=TEST_BUDGET_SECONDS, fail_task_on_error=fail_task_on_error,
        retry_config=RetryConfig(retry_from_step_id="nested", max_retries=2),
    )


def crash_running_tree(database: Path) -> None:
    source = (
        "import asyncio\n"
        "from agent_env.config import configure\n"
        "from agent_env.store import LocalSqliteDocumentStore\n"
        "from agent_env.runner.local_runner import LocalRunner\n"
        "from agent_env.task import Task\n"
        "from tst.unit.task.resource_recovery_test import RecoveryStep,nested_step\n"
        f"configure(document_store=LocalSqliteDocumentStore(path={str(database)!r}))\n"
        "async def main():\n"
        "    leaf=Task(id='leaf',version=1,steps=[RecoveryStep(id='leaf-resource',resource_id='leaf-box',action='crash')])\n"
        "    middle=Task(id='middle',version=1,steps=[RecoveryStep(id='middle-resource',resource_id='middle-box'),nested_step(leaf)])\n"
        "    parent=Task.put(id='crash-parent',steps=[nested_step(middle)])\n"
        "    runner=LocalRunner(workers=1)\n"
        "    await runner.submit(parent.id,parent.version)\n"
        "    await asyncio.Event().wait()\n"
        "asyncio.run(main())\n"
    )
    checkout = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "PYTHONPATH": str(checkout / "src")}
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=checkout, env=environment,
        capture_output=True, text=True, timeout=PROCESS_BUDGET_SECONDS,
    )
    assert result.returncode == 17, result.stderr


@pytest.mark.asyncio
async def test_hard_process_exit_recovers_uncommitted_descendant_resources_from_durable_manifests(recovery_config):
    database, log_path, _ = recovery_config
    await asyncio.to_thread(crash_running_tree, database)
    [record] = run_store.active_runs("local")
    assert not log_path.exists()
    [middle] = get_child_invocation_store().list_for_parent(record.instance_id)
    [leaf] = get_child_invocation_store().list_for_parent(middle.child_instance_id)
    assert leaf.cleanup_context["deployed_sandboxes"][0]["sandbox_id"] == "leaf-box"

    runner = LocalRunner(workers=1)
    try:
        await runner.start()
    finally:
        await runner.stop()

    assert log_path.read_text().splitlines() == ["leaf-box", "middle-box"]
    assert run_store.get_run(record.run_id).status is RunStatus.FAILED
    assert get_child_invocation_store().list_for_parent(record.instance_id)[0].cleanup_state is ChildCleanupState.COMPLETED
    assert get_child_invocation_store().list_for_parent(middle.child_instance_id)[0].cleanup_state is ChildCleanupState.COMPLETED


@pytest.mark.asyncio
async def test_orphan_cleanup_failure_stays_recoverable_until_a_later_start_confirms_termination(recovery_config):
    database, log_path, failure_marker = recovery_config
    await asyncio.to_thread(crash_running_tree, database)
    [record] = run_store.active_runs("local")
    failure_marker.touch()
    runner = LocalRunner(workers=1)
    try:
        await runner.start()
        assert run_store.get_run(record.run_id).status is RunStatus.RUNNING
        failure_marker.unlink()
        await runner.start()
    finally:
        await runner.stop()

    assert run_store.get_run(record.run_id).status is RunStatus.FAILED
    assert log_path.read_text().splitlines() == ["leaf-box", "middle-box"]


@pytest.mark.asyncio
async def test_lifecycle_error_keeps_its_type_and_cannot_be_retried_or_tolerated_by_a_parent(recovery_config):
    child = Task(id="lifecycle-child", version=1, steps=[RecoveryStep(id="failure", action="lifecycle")])
    parent = Task(id="lifecycle-parent", version=1, steps=[nested_step(child, fail_task_on_error=False)])

    with pytest.raises(TaskLifecycleError, match="resource ownership cannot be confirmed") as failure:
        await parent.run()

    assert type(failure.value) is TaskLifecycleError
    assert RecoveryStep.calls["failure"] == 1


@pytest.mark.asyncio
async def test_failed_step_resource_is_cleaned_even_though_its_domain_context_was_never_committed(recovery_config):
    _, log_path, _ = recovery_config
    child = Task(id="failed-child", version=1, steps=[RecoveryStep(id="allocate", resource_id="failed-box", action="fail")])
    parent = Task(id="failed-parent", version=1, steps=[nested_step(child)])
    parent.steps[0].retry_config = None
    context = TaskStepContext()

    with pytest.raises(RuntimeError, match="step failed after recording"):
        await parent.run(context=context)

    assert not context.deployed_sandboxes
    assert log_path.read_text().splitlines() == ["failed-box"]
    [invocation] = get_child_invocation_store().list_for_parent(context.instance_id)
    assert invocation.cleanup_state is ChildCleanupState.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_read", ["children", "inventory"])
async def test_store_discovery_failure_does_not_prevent_known_resource_cleanup(recovery_config, monkeypatch, failed_read):
    _, log_path, _ = recovery_config
    context = TaskStepContext()
    await Task(id="live-owner", version=1, steps=[
        RecoveryStep(id="known", resource_id="known-box"),
    ]).run(context=context)
    documents = get_config().get_document_store()
    query = documents.query
    original = ConnectionError("document store unavailable")

    def query_documents(collection, *args, **kwargs):
        if collection == TASK_CHILD_INVOCATIONS_COLLECTION:
            raise original
        return query(collection, *args, **kwargs)

    async def unavailable_inventory(*args, **kwargs):
        raise original

    with monkeypatch.context() as failure:
        if failed_read == "children":
            failure.setattr(documents, "query", query_documents)
        else:
            failure.setattr(teardown_api, "load_resource_context", unavailable_inventory)
        with pytest.raises(ConnectionError) as caught:
            await teardown_run(context, timeout=TEST_BUDGET_SECONDS)

    assert caught.value is original
    assert log_path.read_text().splitlines() == ["known-box"]
    report = await teardown_run(TaskStepContext(instance_id=context.instance_id), timeout=TEST_BUDGET_SECONDS)
    assert not report.still_up
    assert log_path.read_text().splitlines() == ["known-box"]


@pytest.mark.asyncio
async def test_cleanup_combines_live_and_recovered_resources_without_terminating_either_twice(recovery_config):
    _, log_path, _ = recovery_config
    context = await Task(id="partial-owner", version=1, steps=[
        RecoveryStep(id="known", resource_id="known-box"),
        RecoveryStep(id="recovered", resource_id="recovered-box"),
    ]).run()
    context.deployed_sandboxes = context.deployed_sandboxes[:1]

    report = await teardown_run(context, timeout=TEST_BUDGET_SECONDS)

    assert {sandbox.sandbox_id for sandbox in report.terminated} == {"known-box", "recovered-box"}
    assert not report.still_up
    assert log_path.read_text().splitlines() == ["known-box", "recovered-box"]


@pytest.mark.asyncio
async def test_restored_context_skips_termination_confirmed_only_in_resource_inventory(recovery_config):
    _, log_path, _ = recovery_config
    context = await Task(id="restored-owner", version=1, steps=[
        RecoveryStep(id="known", resource_id="known-box"),
    ]).run()
    instance = get_task_instance_store().get(context.instance_id)
    restored = TaskStepContext.from_dict(instance.context)
    restored.instance_id = context.instance_id

    await teardown_run(context, timeout=TEST_BUDGET_SECONDS)
    report = await teardown_run(restored, timeout=TEST_BUDGET_SECONDS)

    assert not report.still_up
    assert log_path.read_text().splitlines() == ["known-box"]


@pytest.mark.asyncio
async def test_recovered_changelog_is_drained_before_its_committed_resource_is_terminated(recovery_config, monkeypatch):
    _, log_path, _ = recovery_config
    context = await Task(id="checkpointed-changelog", version=1, steps=[
        RecoveryStep(id="agent", resource_id="agent-box"),
    ]).run()
    instance = get_task_instance_store().get(context.instance_id)
    context.metadata["agent_changelog"] = [dict(STAGED_CHANGELOG_ENTRY)]
    await record_resource_context(
        context, attempt_epoch=instance.attempt_epoch,
        deadline=asyncio.get_running_loop().time() + TEST_BUDGET_SECONDS,
    )
    restored = TaskStepContext.from_dict(instance.context)
    restored.instance_id = context.instance_id

    async def collect(namespaces):
        if list(namespaces):
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write("drained-changelog\n")
        return []

    monkeypatch.setattr(teardown_api, "drain_all", collect)
    await teardown_run(restored, timeout=TEST_BUDGET_SECONDS)
    assert log_path.read_text().splitlines() == ["drained-changelog", "agent-box"]


@pytest.mark.asyncio
async def test_cancelled_cleanup_does_not_start_additional_provider_calls(recovery_config, monkeypatch):
    context = await Task(id="cancelled-cleanup", version=1, steps=[
        RecoveryStep(id="known", resource_id="known-box"),
        RecoveryStep(id="recovered", resource_id="recovered-box"),
    ]).run()
    context.deployed_sandboxes = context.deployed_sandboxes[:1]
    entered = asyncio.Event()
    attempts = []

    async def terminate(self, sandbox_id):
        attempts.append(sandbox_id)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(RecoveryProvider, "terminate_sandbox", terminate)
    cleaning = asyncio.create_task(teardown_run(context, timeout=TEST_BUDGET_SECONDS))
    try:
        async with asyncio.timeout(TEST_BUDGET_SECONDS):
            await entered.wait()
        before_cancel = list(attempts)
    finally:
        cleaning.cancel()
        report = await cleaning
    assert attempts == before_cancel
    assert {resource.sandbox_id for resource in report.left} == {"known-box", "recovered-box"}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["changelog", "changelog_fail"])
async def test_child_staged_changelog_is_drained_before_its_sandbox_is_terminated(recovery_config, monkeypatch, action):
    _, log_path, _ = recovery_config
    drained = []

    async def collect(namespaces):
        captured = list(namespaces)
        drained.extend(captured)
        if captured:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write("drained-changelog\n")
        return []

    monkeypatch.setattr("agent_env.task.teardown.drain_all", collect)
    child = Task(id="staged-child", version=1, steps=[RecoveryStep(id="staged", resource_id="staged-box", action=action)])
    parent = Task(id="staged-parent", version=1, steps=[nested_step(child)])
    parent.steps[0].retry_config = None
    context = TaskStepContext()

    if action == "changelog_fail":
        with pytest.raises(RuntimeError, match="step failed after recording"):
            await parent.run(context=context)
    else:
        await parent.run(context=context)

    assert log_path.read_text().splitlines() == ["drained-changelog", "staged-box"]
    assert len(drained) == 1
    assert drained[0].staging_url == STAGED_CHANGELOG_ENTRY["staging_url"]
    assert drained[0].namespace_url == STAGED_CHANGELOG_ENTRY["object_url"]
    assert "agent_changelog" not in context.metadata
    [invocation] = get_child_invocation_store().list_for_parent(context.instance_id)
    assert invocation.cleanup_state is ChildCleanupState.COMPLETED
    assert invocation.cleanup_context["metadata"]["agent_changelog"] == [STAGED_CHANGELOG_ENTRY]
