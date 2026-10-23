"""Nested graphs persist their own execution while publishing only declared outputs."""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.store import LocalFilesystemObjectStore, LocalSqliteDocumentStore
from agent_env.task import Task
from agent_env.task import nested as nested_module
from agent_env.task.child_store import ChildCleanupState, ChildInvocation, get_child_invocation_store
from agent_env.task.nested import ChildCleanupFailure, ChildTaskFailed, ChildTaskTimeout
from agent_env.task.step_journal import commit_ordered, replay_context
from agent_env.task.store import TaskStepStatus, get_task_instance_store
from agent_env.task.teardown import TeardownReport
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps
from agent_env.task_step.task_step import RetryConfig, TaskStep
from agent_env.task_step.task_steps.repeat_until import RepeatUntilExhaustedError, RepeatUntilTaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep


CHILD_TIMEOUT_SECONDS = 5.0
CHILD_CLEANUP_TIMEOUT_SECONDS = 5.0
TEST_DEADLINE_SECONDS = 10.0
SHORT_CHILD_TIMEOUT_SECONDS = 0.05
SHORT_CLEANUP_TIMEOUT_SECONDS = 0.05


@dataclass
class ProbeState:
    calls: Counter[str] = field(default_factory=Counter)
    observed: list[tuple[str, str | None, dict]] = field(default_factory=list)
    entered: dict[str, asyncio.Event] = field(default_factory=dict)
    released: dict[str, asyncio.Event] = field(default_factory=dict)
    cleaned: dict[str, asyncio.Event] = field(default_factory=dict)
    termination_attempts: list[str] = field(default_factory=list)
    termination_fails: bool = False
    crash_after_child: bool = True

    def event(self, events: dict[str, asyncio.Event], name: str) -> asyncio.Event:
        return events.setdefault(name, asyncio.Event())


class ProbeStep(TaskStep):
    type = "nested_execution_probe"
    entity_refs = ()
    state = ProbeState()

    def __init__(
        self, *, id: str, version: int | None = None, action: str = "write", key: str = "answer",
        value: str | int | bool | dict | None = None, target: int = 1, depends_on: list | None = None,
        fail_task_on_error: bool = True, retry_config: RetryConfig | dict | None = None,
    ) -> None:
        super().__init__(
            id=id, version=version, depends_on=depends_on,
            fail_task_on_error=fail_task_on_error, retry_config=retry_config,
        )
        self.action = action
        self.key = key
        self.value = value
        self.target = target

    def to_dict(self) -> dict:
        return {**super().to_dict(), "action": self.action, "key": self.key, "value": self.value, "target": self.target}

    @classmethod
    def from_dict(cls, data: dict) -> ProbeStep:
        return cls(
            **cls._base_from_dict(data), action=data["action"], key=data["key"],
            value=data["value"], target=data["target"], retry_config=data.get("retry_config"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        state = self.state
        state.calls[self.id] += 1
        state.observed.append((self.id, context.instance_id, copy.deepcopy(context.metadata)))
        if self.action == "echo":
            context.metadata["payload"]["child_mutation"] = self.id
            context.metadata[self.key] = copy.deepcopy(context.metadata["payload"])
            context.metadata["child_only"] = True
        elif self.action == "increment":
            context.metadata[self.key] += 1
            context.metadata["done"] = context.metadata[self.key] >= self.target
        elif self.action == "increment_then_fail_once":
            context.metadata[self.key] += 1
            if state.calls[self.id] == 1:
                raise RuntimeError("child failed after changing uncommitted context")
        elif self.action == "resource":
            resource_id = f"{self.id}-{state.calls[self.id]}"
            context.deployed_sandboxes.append(DeployedSandbox(
                sandbox_name=self.id, sandbox_id=resource_id,
                sandbox_mode="container", sandbox_type="nested-test-provider",
            ))
            context.metadata[self.key] = resource_id
        elif self.action == "fail_after_child_started":
            await state.event(state.entered, "blocking").wait()
            raise RuntimeError("independent parent branch failed")
        elif self.action == "fail" or self.action == "fail_once" and state.calls[self.id] == 1:
            raise RuntimeError(f"probe failure: {self.id}")
        elif self.action in ("wait", "wait_once"):
            state.event(state.entered, self.id).set()
            try:
                if self.action == "wait" or state.calls[self.id] == 1:
                    await state.event(state.released, self.id).wait()
            finally:
                state.event(state.cleaned, self.id).set()
            context.metadata[self.key] = copy.deepcopy(self.value)
        elif self.action == "signal":
            state.event(state.entered, self.id).set()
        else:
            context.metadata[self.key] = copy.deepcopy(self.value)
        return context


class CrashAfterChildStep(SubDAGTaskStep):
    entity_refs = ()

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        await super().execute(context)
        if ProbeStep.state.crash_after_child:
            ProbeStep.state.crash_after_child = False
            raise RuntimeError("parent interrupted after child completed")
        return context


class CleanupProvider(SandboxProvider):
    state = ProbeState()

    async def create_sandbox(self, **parameters) -> Sandbox:
        raise AssertionError("the cleanup fixture never provisions a real sandbox")

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        raise AssertionError("native teardown must terminate the recorded sandbox by id")

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        self.state.termination_attempts.append(sandbox_id)
        if self.state.termination_fails:
            raise RuntimeError("sandbox termination was not confirmed")


@pytest.fixture
def nested_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProbeState:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[task_steps]\nimpls = ["tst.unit.task.nested_task_execution_test:ProbeStep"]\n'
        '[sandbox.providers.nested-test-provider]\n'
        'impl = "tst.unit.task.nested_task_execution_test:CleanupProvider"\n', encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config_path))
    configure(
        document_store=LocalSqliteDocumentStore(path=str(tmp_path / "documents.sqlite3")),
        object_store=LocalFilesystemObjectStore(root=str(tmp_path / "objects")),
    )
    state = ProbeState()
    monkeypatch.setattr(ProbeStep, "state", state)
    monkeypatch.setattr(CleanupProvider, "state", state)
    yield state
    reset_config()


def child_task(*steps: TaskStep) -> Task:
    return Task(id="nested-child", version=1, steps=list(steps))


def subdag_step(task: Task, *, id: str = "nested", **bindings) -> SubDAGTaskStep:
    return SubDAGTaskStep(id=id, version=None, task=task, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
        cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS, **bindings,
    )


async def resume_parent(task: Task, instance_id: str) -> TaskStepContext:
    previous = get_task_instance_store().get(instance_id)
    completed = {step.step_id for step in previous.completed_steps if step.status == TaskStepStatus.SUCCESS}
    return await task.run(
        instance_id=instance_id, context=TaskStepContext.from_dict(previous.context), completed_step_ids=completed,
    )


@pytest.mark.asyncio
async def test_subdag_isolates_parent_context_and_records_child_output_handoff(nested_state: ProbeState) -> None:
    child = child_task(ProbeStep(id="echo", action="echo"))
    task = Task(id="isolated-parent", version=1, steps=[
        subdag_step(child, input_bindings={"payload": "source"}, output_bindings={"result": "answer"}),
    ])
    context = TaskStepContext(metadata={"source": {"input": "original"}, "private_parent": "not passed"})

    result = await task.run(context=context)

    assert result.metadata["source"] == {"input": "original"}
    assert result.metadata["result"] == {"input": "original", "child_mutation": "echo"}
    assert "child_only" not in result.metadata
    assert "private_parent" not in nested_state.observed[0][2]
    invocations = get_child_invocation_store().list_for_parent(result.instance_id)
    assert len(invocations) == 1
    invocation = invocations[0]
    child_instance = get_task_instance_store().get(invocation.child_instance_id)
    assert child_instance.status == "completed"
    assert child_instance.context["metadata"]["child_only"] is True
    assert invocation.child_instance_id != result.instance_id
    assert nested_state.observed[0][1] == invocation.child_instance_id
    parent_instance = get_task_instance_store().get(result.instance_id)
    assert [(step.step_id, step.status) for step in parent_instance.completed_steps] == [("nested", TaskStepStatus.SUCCESS)]
    assert invocation.task_definition == child.to_dict()
    assert invocation.captured_inputs["context"]["metadata"]["payload"] == {"input": "original"}
    assert invocation.captured_inputs["child_timeout_seconds"] == CHILD_TIMEOUT_SECONDS
    assert invocation.captured_inputs["cleanup_timeout_seconds"] == CHILD_CLEANUP_TIMEOUT_SECONDS
    parent_journal = get_task_instance_store().journal_entries_sync(result.instance_id)
    completed_entry = next(entry for entry in parent_journal if entry["step_id"] == "nested")
    changes = ContextUpdateOps.from_journal_dict(completed_entry["ops"])
    replayed = replay_context(commit_ordered(parent_journal, [
        {"step_id": step.step_id, "status": step.status} for step in parent_instance.completed_steps
    ]), ())
    assert replayed["metadata"]["result"] == {"input": "original", "child_mutation": "echo"}
    assert any(path.startswith("context.metadata.result") for path in changes.sets)
    assert not any("child_only" in path or "payload" in path for path in changes.sets)


@pytest.mark.asyncio
async def test_completed_child_is_reused_after_parent_fails_before_its_step_completion(nested_state: ProbeState) -> None:
    nested = CrashAfterChildStep(
        id="nested", version=None, task=child_task(ProbeStep(id="child-write", value="durable")),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        output_bindings={"result": "answer"},
    )
    task = Task(id="crashed-parent", version=1, steps=[nested])
    context = TaskStepContext()
    with pytest.raises(RuntimeError, match="parent interrupted after child completed"):
        await task.run(context=context)
    first = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(first) == 1
    assert get_task_instance_store().get(first[0].child_instance_id).status == "completed"

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "durable"
    assert nested_state.calls["child-write"] == 1
    assert [item.child_instance_id for item in get_child_invocation_store().list_for_parent(context.instance_id)] == [
        first[0].child_instance_id,
    ]
    assert get_task_instance_store().get(context.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_resuming_midchild_preserves_completed_independent_branch_after_an_earlier_hole(
    nested_state: ProbeState,
) -> None:
    child = child_task(
        ProbeStep(id="earlier-hole", action="wait_once", value="resumed", depends_on=[]),
        ProbeStep(id="completed-branch", key="branch", value="kept", depends_on=[]),
        ProbeStep(id="branch-committed", action="signal", depends_on=["completed-branch"]),
        ProbeStep(id="join", value="joined", depends_on=["earlier-hole", "branch-committed"]),
    )
    task = Task(id="interrupted-parent", version=1, steps=[subdag_step(child, output_bindings={"result": "answer"})])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "earlier-hole").wait()
            await nested_state.event(nested_state.entered, "branch-committed").wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    interrupted = get_task_instance_store().get(invocation.child_instance_id)
    assert "completed-branch" in {step.step_id for step in interrupted.completed_steps}
    assert "earlier-hole" not in {step.step_id for step in interrupted.completed_steps}

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "joined"
    assert nested_state.calls["earlier-hole"] == 2
    assert nested_state.calls["completed-branch"] == 1
    assert nested_state.calls["join"] == 1
    assert len(get_child_invocation_store().list_for_parent(context.instance_id)) == 1


@pytest.mark.asyncio
async def test_parent_retry_rollback_creates_a_fresh_child_invocation(nested_state: ProbeState) -> None:
    nested = subdag_step(child_task(ProbeStep(id="child-run", value="written")), output_bindings={"result": "answer"})
    fail_once = ProbeStep(
        id="retry-parent", action="fail_once", retry_config=RetryConfig(retry_from_step_id="nested", max_retries=1),
    )
    task = Task(id="retrying-parent", version=1, steps=[nested, fail_once])

    result = await task.run()

    invocations = get_child_invocation_store().list_for_parent(result.instance_id)
    assert nested_state.calls["child-run"] == 2
    assert nested_state.calls["retry-parent"] == 2
    assert len(invocations) == 2
    assert len({item.child_instance_id for item in invocations}) == 2
    assert {item.parent_step_generation for item in invocations} == {0, 1}
    assert all(get_task_instance_store().get(item.child_instance_id).status == "completed" for item in invocations)
    assert get_task_instance_store().get(result.instance_id).attempt_epoch == 1


@pytest.mark.asyncio
async def test_repeat_until_feeds_back_isolated_outputs_and_stops_at_convergence(nested_state: ProbeState) -> None:
    repeated = RepeatUntilTaskStep(
        id="repeat", version=None, task=child_task(ProbeStep(id="advance", action="increment", key="count", target=3)),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        max_iterations=5, until_key="done", until_value=True,
        input_bindings={"count": "initial"}, feedback_bindings={"count": "count"}, output_bindings={"result": "count"},
    )
    task = Task(id="converging-parent", version=1, steps=[repeated])

    result = await task.run(context=TaskStepContext(metadata={"initial": 0}))

    assert result.metadata["initial"] == 0
    assert result.metadata["result"] == 3
    assert "count" not in result.metadata and "done" not in result.metadata
    assert nested_state.calls["advance"] == 3
    invocations = get_child_invocation_store().list_for_parent(result.instance_id)
    assert [item.iteration for item in invocations] == [0, 1, 2]
    assert [item.captured_inputs["context"]["metadata"]["count"] for item in invocations] == [0, 1, 2]
    assert len({item.child_instance_id for item in invocations}) == 3


@pytest.mark.asyncio
async def test_repeat_until_exhaustion_never_publishes_partial_outputs(nested_state: ProbeState) -> None:
    repeated = RepeatUntilTaskStep(
        id="repeat", version=None, task=child_task(ProbeStep(id="advance", action="increment", key="count", target=3)),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        max_iterations=2, until_key="done", until_value=True,
        input_bindings={"count": "initial"}, feedback_bindings={"count": "count"}, output_bindings={"result": "count"},
    )
    task = Task(id="exhausted-parent", version=1, steps=[repeated])
    context = TaskStepContext(metadata={"initial": 0, "result": "unchanged"})

    with pytest.raises(RepeatUntilExhaustedError):
        await task.run(context=context)

    assert context.metadata["result"] == "unchanged"
    assert nested_state.calls["advance"] == 2
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    assert [item.iteration for item in invocations] == [0, 1]
    assert all(get_task_instance_store().get(item.child_instance_id).status == "completed" for item in invocations)
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_tolerated_child_failure_propagates_without_publishing_successor_outputs(nested_state: ProbeState) -> None:
    child = child_task(
        ProbeStep(id="child-failure", action="fail", fail_task_on_error=False),
        ProbeStep(id="child-successor", value="must not publish"),
    )
    task = Task(id="failing-parent", version=1, steps=[subdag_step(child, output_bindings={"result": "answer"})])
    context = TaskStepContext(metadata={"result": "unchanged"})

    with pytest.raises(ChildTaskFailed) as failed:
        await task.run(context=context)

    assert nested_state.calls["child-successor"] == 1
    assert context.metadata["result"] == "unchanged"
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert failed.value.child_instance_id == invocation.child_instance_id
    child_instance = get_task_instance_store().get(invocation.child_instance_id)
    assert child_instance.context["metadata"]["failed_steps"][0]["step_id"] == "child-failure"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_parent_cancellation_drains_running_child_and_records_cleanup(nested_state: ProbeState) -> None:
    task = Task(id="cancelled-parent", version=1, steps=[subdag_step(child_task(ProbeStep(id="blocking", action="wait")))])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "blocking").wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert nested_state.event(nested_state.cleaned, "blocking").is_set()
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.cleanup_state == "completed"
    assert get_task_instance_store().get(invocation.child_instance_id).status == "cancelled"
    assert get_task_instance_store().get(context.instance_id).status == "cancelled"


@pytest.mark.asyncio
async def test_child_deadline_cancels_the_running_graph_without_publishing_output(nested_state: ProbeState) -> None:
    nested = SubDAGTaskStep(
        id="nested", version=None, task=child_task(ProbeStep(id="blocking", action="wait")),
        child_timeout_seconds=SHORT_CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        output_bindings={"result": "answer"},
    )
    task = Task(id="timed-out-parent", version=1, steps=[nested])
    context = TaskStepContext(metadata={"result": "unchanged"})

    with pytest.raises(ChildTaskTimeout):
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await task.run(context=context)

    assert context.metadata["result"] == "unchanged"
    assert nested_state.event(nested_state.entered, "blocking").is_set()
    assert nested_state.event(nested_state.cleaned, "blocking").is_set()
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.cleanup_state == "completed"
    assert get_task_instance_store().get(invocation.child_instance_id).status == "cancelled"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("child_fails", (False, True))
@pytest.mark.parametrize("parent_policy", ("fatal", "tolerant", "retry"))
async def test_cleanup_deadline_is_bounded_and_takes_priority_over_child_outcome(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch, child_fails: bool, parent_policy: str,
) -> None:
    cleanup_entered = asyncio.Event()
    cleanup_drained = asyncio.Event()

    async def blocked_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        cleanup_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_drained.set()
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", blocked_teardown)
    nested = SubDAGTaskStep(
        id="nested", version=None,
        task=child_task(ProbeStep(id="child-outcome", action="fail" if child_fails else "write", value="unpublished")),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=SHORT_CLEANUP_TIMEOUT_SECONDS,
        output_bindings={"result": "answer"},
        fail_task_on_error=parent_policy != "tolerant",
        retry_config=RetryConfig(retry_from_step_id="nested", max_retries=1) if parent_policy == "retry" else None,
    )
    task = Task(id="cleanup-deadline-parent", version=1, steps=[nested])
    context = TaskStepContext(metadata={"result": "unchanged"})

    with pytest.raises(ChildCleanupFailure):
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await task.run(context=context)

    assert cleanup_entered.is_set() and cleanup_drained.is_set()
    assert context.metadata["result"] == "unchanged"
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.cleanup_state == "failed"
    assert invocation.cleanup_error is not None
    assert context.metadata["failed_steps"][-1]["error_type"] == "ChildCleanupFailure"
    assert get_task_instance_store().get(context.instance_id).status == "failed"
    assert nested_state.calls["child-outcome"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("completed", "problem"), (
    ({"missing"}, "outside the active task"),
    ({"last"}, "unfinished dependencies"),
))
async def test_completed_step_ids_reject_unknown_steps_and_dependency_holes_before_running(
    nested_state: ProbeState, completed: set[str], problem: str,
) -> None:
    task = Task(id="invalid-resume", version=1, steps=[ProbeStep(id="first"), ProbeStep(id="last")])
    context = TaskStepContext()

    with pytest.raises(ValueError, match=problem):
        await task.run(context=context, completed_step_ids=completed)

    assert not nested_state.calls
    assert context.instance_id is None


@pytest.mark.asyncio
async def test_partial_child_redeploys_resources_that_prior_cancellation_already_cleaned(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned_sandboxes: list[list[str]] = []

    async def record_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        cleaned_sandboxes.append([sandbox.sandbox_id for sandbox in context.deployed_sandboxes])
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", record_teardown)
    child = child_task(
        ProbeStep(id="deploy", action="resource"),
        ProbeStep(id="use-resource", action="wait_once", value="done"),
    )
    task = Task(id="resource-resume-parent", version=1, steps=[subdag_step(child, output_bindings={"result": "answer"})])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "use-resource").wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    first_invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert first_invocation.cleanup_state == "completed"
    assert cleaned_sandboxes == [["deploy-1"]]

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "done"
    assert not resumed.deployed_sandboxes
    assert nested_state.calls["deploy"] == nested_state.calls["use-resource"] == 2
    assert cleaned_sandboxes == [["deploy-1"], ["deploy-2"]]
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(invocations) == 1
    assert invocations[0].child_instance_id == first_invocation.child_instance_id


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_source", ("parent_cancellation", "sibling_failure"))
async def test_child_cleanup_failure_survives_scheduler_drain_after_parent_stops(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch, stop_source: str,
) -> None:
    cleanup_drained = asyncio.Event()

    async def blocked_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_drained.set()
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", blocked_teardown)
    nested = SubDAGTaskStep(
        id="nested", version=None, task=child_task(ProbeStep(id="blocking", action="wait")),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=SHORT_CLEANUP_TIMEOUT_SECONDS,
        depends_on=[],
    )
    steps = [nested]
    if stop_source == "sibling_failure":
        steps.append(ProbeStep(id="parent-failure", action="fail_after_child_started", depends_on=[]))
    task = Task(id="draining-parent", version=1, steps=steps)
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "blocking").wait()
        if stop_source == "parent_cancellation":
            running.cancel()
        with pytest.raises(ChildCleanupFailure):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert cleanup_drained.is_set()
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.cleanup_state == "failed"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_cancellation_during_successful_child_cleanup_finishes_it_and_resume_reuses_child(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_drained = asyncio.Event()
    cleaned_instances: list[str | None] = []

    async def controlled_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        cleaned_instances.append(context.instance_id)
        cleanup_entered.set()
        try:
            await cleanup_release.wait()
        finally:
            cleanup_drained.set()
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", controlled_teardown)
    task = Task(id="cleanup-cancel-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="completed-before-cancel", value="durable")), output_bindings={"result": "answer"},
    )])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await cleanup_entered.wait()
        running.cancel()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        cleanup_release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.outcome is not None
    assert invocation.cleanup_state == "completed"
    assert cleanup_drained.is_set()
    assert "result" not in context.metadata

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "durable"
    assert nested_state.calls["completed-before-cancel"] == 1
    assert cleaned_instances == [invocation.child_instance_id]


@pytest.mark.asyncio
async def test_failed_cleanup_resume_keeps_recorded_resources_until_termination_is_confirmed(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_sandboxes: list[list[str]] = []

    async def failed_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        attempted_sandboxes.append([sandbox.sandbox_id for sandbox in context.deployed_sandboxes])
        raise RuntimeError("resource termination was not confirmed")

    monkeypatch.setattr(nested_module, "teardown_run", failed_teardown)
    task = Task(id="cleanup-recovery-parent", version=1, steps=[subdag_step(child_task(
        ProbeStep(id="deploy", action="resource"), ProbeStep(id="child-failure", action="fail"),
    ))])
    context = TaskStepContext()
    with pytest.raises(ChildCleanupFailure):
        await task.run(context=context)
    original = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert attempted_sandboxes == [["deploy-1"]]
    assert original.cleanup_state == "failed"

    with pytest.raises(ChildCleanupFailure):
        await resume_parent(task, context.instance_id)

    assert attempted_sandboxes == [["deploy-1"], ["deploy-1"]]
    assert nested_state.calls["deploy"] == nested_state.calls["child-failure"] == 1
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.child_instance_id == original.child_instance_id
    assert invocation.cleanup_state == "failed"


@pytest.mark.asyncio
async def test_completed_child_instance_is_recovered_after_its_ledger_outcome_write_fails(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = get_child_invocation_store()
    complete = store.complete
    attempted_completions: list[str] = []

    def fail_first_completion(invocation: ChildInvocation, *, outcome: dict) -> ChildInvocation:
        attempted_completions.append(invocation.child_instance_id)
        if len(attempted_completions) == 1:
            raise OSError("child outcome write interrupted")
        return complete(invocation, outcome=outcome)

    monkeypatch.setattr(store, "complete", fail_first_completion)
    task = Task(id="missing-ledger-outcome-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="completed-child", value="durable")), output_bindings={"result": "answer"},
    )])
    context = TaskStepContext()

    with pytest.raises(ChildTaskFailed) as failure:
        await task.run(context=context)

    original = store.list_for_parent(context.instance_id)[0]
    assert isinstance(failure.value.__cause__, OSError)
    assert original.outcome is None
    assert original.cleanup_state == "completed"
    assert get_task_instance_store().get(original.child_instance_id).status == "completed"
    assert "result" not in context.metadata

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "durable"
    assert nested_state.calls["completed-child"] == 1
    assert attempted_completions == [original.child_instance_id, original.child_instance_id]
    recovered = store.list_for_parent(context.instance_id)
    assert len(recovered) == 1
    assert recovered[0].child_instance_id == original.child_instance_id
    assert recovered[0].outcome["metadata"]["answer"] == "durable"


@pytest.mark.asyncio
async def test_two_level_subdags_record_immediate_parent_links_and_handoff_each_output(
    nested_state: ProbeState,
) -> None:
    leaf = Task(id="leaf-task", version=1, steps=[ProbeStep(id="leaf-echo", action="echo")])
    middle = Task(id="middle-task", version=1, steps=[subdag_step(
        leaf, id="inner", input_bindings={"payload": "payload"}, output_bindings={"middle_result": "answer"},
    )])
    task = Task(id="two-level-parent", version=1, steps=[subdag_step(
        middle, id="outer", input_bindings={"payload": "source"}, output_bindings={"result": "middle_result"},
    )])
    context = TaskStepContext(metadata={"source": {"value": "root"}, "private_parent": "not inherited"})

    result = await task.run(context=context)

    outer_invocations = get_child_invocation_store().list_for_parent(result.instance_id)
    assert len(outer_invocations) == 1
    outer = outer_invocations[0]
    inner_invocations = get_child_invocation_store().list_for_parent(outer.child_instance_id)
    assert len(inner_invocations) == 1
    inner = inner_invocations[0]
    middle_instance = get_task_instance_store().get(outer.child_instance_id)
    leaf_instance = get_task_instance_store().get(inner.child_instance_id)
    assert len({result.instance_id, middle_instance.instance_id, leaf_instance.instance_id}) == 3
    assert outer.parent_step_id == "outer" and inner.parent_step_id == "inner"
    assert middle_instance.task_id == "middle-task" and leaf_instance.task_id == "leaf-task"
    assert middle_instance.status == leaf_instance.status == "completed"
    assert middle_instance.context["metadata"]["parent_task"]["instance_id"] == result.instance_id
    assert leaf_instance.context["metadata"]["parent_task"]["instance_id"] == middle_instance.instance_id
    assert leaf_instance.context["metadata"]["parent_task"]["step_id"] == "inner"
    assert result.metadata["source"] == {"value": "root"}
    assert result.metadata["result"] == {"value": "root", "child_mutation": "leaf-echo"}
    assert "private_parent" not in middle_instance.context["metadata"]
    assert "private_parent" not in leaf_instance.context["metadata"]
    assert "middle_result" not in result.metadata
    assert "child_only" not in middle_instance.context["metadata"]
    assert get_child_invocation_store().list_for_parent(leaf_instance.instance_id) == []
    assert outer.cleanup_state == inner.cleanup_state == "completed"
    assert nested_state.calls["leaf-echo"] == 1


@pytest.mark.asyncio
async def test_nested_repeat_until_resume_reuses_prior_iterations_after_interrupted_ledger_write(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = get_child_invocation_store()
    complete = store.complete
    interrupted = False

    def interrupt_second_iteration(invocation: ChildInvocation, *, outcome: dict) -> ChildInvocation:
        nonlocal interrupted
        if invocation.parent_step_id == "repeat" and invocation.iteration == 1 and not interrupted:
            interrupted = True
            raise OSError("second iteration outcome write interrupted")
        return complete(invocation, outcome=outcome)

    monkeypatch.setattr(store, "complete", interrupt_second_iteration)
    repeated = RepeatUntilTaskStep(
        id="repeat", version=None, task=child_task(ProbeStep(id="advance", action="increment", key="count", target=3)),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        max_iterations=5, until_key="done", until_value=True,
        input_bindings={"count": "initial"}, feedback_bindings={"count": "count"}, output_bindings={"answer": "count"},
    )
    middle = Task(id="loop-task", version=1, steps=[repeated])
    task = Task(id="nested-loop-resume-parent", version=1, steps=[subdag_step(
        middle, input_bindings={"initial": "initial"}, output_bindings={"result": "answer"},
    )])
    context = TaskStepContext(metadata={"initial": 0})

    with pytest.raises(ChildTaskFailed):
        await task.run(context=context)

    outer = store.list_for_parent(context.instance_id)[0]
    previous_iterations = store.list_for_parent(outer.child_instance_id)
    assert [item.iteration for item in previous_iterations] == [0, 1]
    assert previous_iterations[0].outcome is not None
    assert previous_iterations[1].outcome is None
    assert all(get_task_instance_store().get(item.child_instance_id).status == "completed" for item in previous_iterations)
    assert nested_state.calls["advance"] == 2
    assert "result" not in context.metadata

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == 3
    assert nested_state.calls["advance"] == 3
    recovered_outer = store.list_for_parent(context.instance_id)
    assert len(recovered_outer) == 1
    assert recovered_outer[0].child_instance_id == outer.child_instance_id
    recovered_iterations = store.list_for_parent(outer.child_instance_id)
    assert [item.iteration for item in recovered_iterations] == [0, 1, 2]
    assert [item.child_instance_id for item in recovered_iterations[:2]] == [
        item.child_instance_id for item in previous_iterations
    ]
    assert [metadata["count"] for step, _, metadata in nested_state.observed if step == "advance"] == [0, 1, 2]
    assert all(item.cleanup_state == "completed" for item in recovered_iterations)


@pytest.mark.asyncio
async def test_restarting_a_cleaned_resource_child_invalidates_its_completed_grandchild_invocation(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned_sandboxes: list[list[str]] = []

    async def record_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        cleaned_sandboxes.append([sandbox.sandbox_id for sandbox in context.deployed_sandboxes])
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", record_teardown)
    grandchild = Task(id="resource-grandchild", version=1, steps=[ProbeStep(id="grandchild-call", value="checked")])
    middle = Task(id="resource-child", version=1, steps=[
        ProbeStep(id="deploy", action="resource", key="resource_id"),
        subdag_step(
            grandchild, id="grandchild", input_bindings={"resource_id": "resource_id"},
            output_bindings={"nested_answer": "answer"},
        ),
        ProbeStep(id="use-resource", action="wait_once", value="done"),
    ])
    task = Task(id="resource-grandchild-parent", version=1, steps=[subdag_step(
        middle, output_bindings={"result": "answer"},
    )])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "use-resource").wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    outer = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    original_grandchildren = get_child_invocation_store().list_for_parent(outer.child_instance_id)
    assert len(original_grandchildren) == 1
    assert original_grandchildren[0].captured_inputs["context"]["metadata"]["resource_id"] == "deploy-1"

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == "done"
    assert not resumed.deployed_sandboxes
    assert nested_state.calls["deploy"] == nested_state.calls["grandchild-call"] == 2
    recovered_outer = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(recovered_outer) == 1
    assert recovered_outer[0].child_instance_id == outer.child_instance_id
    grandchildren = get_child_invocation_store().list_for_parent(outer.child_instance_id)
    assert len(grandchildren) == 2
    assert {item.parent_step_generation for item in grandchildren} == {0, 1}
    assert len({item.child_instance_id for item in grandchildren}) == 2
    assert [metadata["resource_id"] for step, _, metadata in nested_state.observed if step == "grandchild-call"] == [
        "deploy-1", "deploy-2",
    ]
    assert [sandboxes for sandboxes in cleaned_sandboxes if sandboxes] == [["deploy-1"], ["deploy-2"]]


@pytest.mark.asyncio
async def test_repeated_parent_cancellation_drains_child_cleanup_and_records_terminal_states(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_drained = asyncio.Event()

    async def controlled_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        cleanup_entered.set()
        try:
            await cleanup_release.wait()
        finally:
            cleanup_drained.set()
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", controlled_teardown)
    task = Task(id="repeated-cancel-parent", version=1, steps=[subdag_step(child_task(ProbeStep(
        id="blocking", action="wait",
    )))])
    context = TaskStepContext()
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await nested_state.event(nested_state.entered, "blocking").wait()
        running.cancel()
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await cleanup_entered.wait()
        running.cancel()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await running
    finally:
        cleanup_release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert cleanup_drained.is_set()
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.outcome is None
    assert invocation.cleanup_state == "completed"
    assert get_task_instance_store().get(invocation.child_instance_id).status == "cancelled"
    assert get_task_instance_store().get(context.instance_id).status == "cancelled"


@pytest.mark.asyncio
async def test_resume_with_new_runner_trace_ids_reuses_the_completed_child(nested_state: ProbeState) -> None:
    task = Task(id="new-runner-trace-parent", version=1, steps=[CrashAfterChildStep(
        id="nested", version=None, task=child_task(ProbeStep(id="child-write", value="durable")),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CHILD_CLEANUP_TIMEOUT_SECONDS,
        input_bindings={"input": "input"}, output_bindings={"result": "answer"},
    )])
    context = TaskStepContext(metadata={"input": "unchanged", "workflow_id": "first-run", "run_group_id": "first-group"})
    with pytest.raises(RuntimeError, match="parent interrupted after child completed"):
        await task.run(context=context)
    original = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    previous = get_task_instance_store().get(context.instance_id)
    restored = TaskStepContext.from_dict(previous.context)
    restored.metadata.update(workflow_id="resumed-run", run_group_id="resumed-group")
    completed = {step.step_id for step in previous.completed_steps if step.status == TaskStepStatus.SUCCESS}

    resumed = await task.run(context=restored, instance_id=context.instance_id, completed_step_ids=completed)

    assert resumed.metadata["result"] == "durable"
    assert resumed.metadata["workflow_id"] == "resumed-run"
    assert resumed.metadata["run_group_id"] == "resumed-group"
    assert nested_state.calls["child-write"] == 1
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(invocations) == 1
    assert invocations[0].child_instance_id == original.child_instance_id


@pytest.mark.asyncio
async def test_failed_running_cleanup_write_still_terminates_resources_and_can_confirm_completion(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = get_child_invocation_store()
    record_cleanup = store.set_cleanup_state
    attempted_states: list[ChildCleanupState] = []

    def fail_first_running_write(
        invocation: ChildInvocation, *, state: ChildCleanupState, error: str | None = None,
    ) -> ChildInvocation:
        attempted_states.append(state)
        if state is ChildCleanupState.RUNNING and attempted_states.count(state) == 1:
            raise OSError("running cleanup state write interrupted")
        return record_cleanup(invocation, state=state, error=error)

    monkeypatch.setattr(store, "set_cleanup_state", fail_first_running_write)
    task = Task(id="running-write-fault-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="resource", action="resource", key="resource_id")),
        output_bindings={"result": "resource_id"},
    )])

    result = await task.run()

    assert result.metadata["result"] == "resource-1"
    assert nested_state.termination_attempts == ["resource-1"]
    assert attempted_states.count(ChildCleanupState.RUNNING) == 1
    assert attempted_states.count(ChildCleanupState.COMPLETED) == 1
    invocation = store.list_for_parent(result.instance_id)[0]
    assert invocation.cleanup_state is ChildCleanupState.COMPLETED
    assert get_task_instance_store().get(result.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_teardown_failure_remains_fatal_when_recording_failed_cleanup_also_fails(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_state.termination_fails = True
    store = get_child_invocation_store()
    record_cleanup = store.set_cleanup_state
    attempted_states: list[ChildCleanupState] = []

    def fail_failed_cleanup_write(
        invocation: ChildInvocation, *, state: ChildCleanupState, error: str | None = None,
    ) -> ChildInvocation:
        attempted_states.append(state)
        if state is ChildCleanupState.FAILED:
            raise OSError("failed cleanup state write interrupted")
        return record_cleanup(invocation, state=state, error=error)

    monkeypatch.setattr(store, "set_cleanup_state", fail_failed_cleanup_write)
    task = Task(id="failed-write-fault-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="resource", action="resource", key="resource_id")),
        output_bindings={"result": "resource_id"}, fail_task_on_error=False,
    )])
    context = TaskStepContext()

    with pytest.raises(ChildCleanupFailure):
        await task.run(context=context)

    assert nested_state.termination_attempts == ["resource-1"]
    assert attempted_states.count(ChildCleanupState.FAILED) == 1
    assert "result" not in context.metadata
    invocation = store.list_for_parent(context.instance_id)[0]
    assert invocation.cleanup_state is ChildCleanupState.RUNNING
    assert context.metadata["failed_steps"][-1]["error_type"] == "ChildCleanupFailure"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_missing_completed_cleanup_record_stays_fatal_even_after_resources_are_terminated(
    nested_state: ProbeState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = get_child_invocation_store()
    record_cleanup = store.set_cleanup_state
    attempted_states: list[ChildCleanupState] = []

    def fail_completed_cleanup_write(
        invocation: ChildInvocation, *, state: ChildCleanupState, error: str | None = None,
    ) -> ChildInvocation:
        attempted_states.append(state)
        if state is ChildCleanupState.COMPLETED:
            raise OSError("completed cleanup state write interrupted")
        return record_cleanup(invocation, state=state, error=error)

    monkeypatch.setattr(store, "set_cleanup_state", fail_completed_cleanup_write)
    task = Task(id="completed-write-fault-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="resource", action="resource", key="resource_id")),
        output_bindings={"result": "resource_id"}, fail_task_on_error=False,
    )])
    context = TaskStepContext()

    with pytest.raises(ChildCleanupFailure):
        await task.run(context=context)

    assert nested_state.termination_attempts == ["resource-1"]
    assert attempted_states.count(ChildCleanupState.COMPLETED) == 1
    assert "result" not in context.metadata
    invocation = store.list_for_parent(context.instance_id)[0]
    assert invocation.outcome is not None
    assert invocation.cleanup_state is not ChildCleanupState.COMPLETED
    assert context.metadata["failed_steps"][-1]["error_type"] == "ChildCleanupFailure"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_failed_child_mutations_are_not_replayed_as_completed_work_on_resume(nested_state: ProbeState) -> None:
    task = Task(id="dirty-child-context-parent", version=1, steps=[subdag_step(
        child_task(ProbeStep(id="increment-failure", action="increment_then_fail_once", key="count")),
        input_bindings={"count": "initial"}, output_bindings={"result": "count"},
    )])
    context = TaskStepContext(metadata={"initial": 0})

    with pytest.raises(ChildTaskFailed):
        await task.run(context=context)

    original = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    failed_child = get_task_instance_store().get(original.child_instance_id)
    assert failed_child.context["metadata"]["count"] == 1
    assert original.outcome is None
    assert "result" not in context.metadata

    resumed = await resume_parent(task, context.instance_id)

    assert resumed.metadata["result"] == 1
    assert nested_state.calls["increment-failure"] == 2
    assert [metadata["count"] for step, _, metadata in nested_state.observed if step == "increment-failure"] == [0, 0]
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(invocations) == 1
    assert invocations[0].child_instance_id == original.child_instance_id


@pytest.mark.asyncio
async def test_all_completed_step_ids_skip_dispatch_and_persist_a_terminal_completed_instance(
    nested_state: ProbeState,
) -> None:
    task = Task(id="already-complete-task", version=1, steps=[ProbeStep(id="first"), ProbeStep(id="last")])
    context = TaskStepContext(metadata={"first_effect": "retained", "last_effect": "retained"})

    result = await task.run(context=context, completed_step_ids={"first", "last"})

    assert not nested_state.calls
    assert result.metadata["first_effect"] == result.metadata["last_effect"] == "retained"
    instance = get_task_instance_store().get(result.instance_id)
    assert instance.status == "completed"
    assert instance.current_step == instance.total_steps == 2
    assert instance.completed_at_utc is not None
    assert {step.step_id for step in instance.completed_steps} == {"first", "last"}
    assert all(step.status is TaskStepStatus.SUCCESS for step in instance.completed_steps)
    assert instance.context["metadata"]["first_effect"] == instance.context["metadata"]["last_effect"] == "retained"
