"""Required persistence prevents the DAG from publishing work the store could not commit."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest

from agent_env.config import configure, reset_config
from agent_env.store import LocalFilesystemObjectStore, LocalSqliteDocumentStore
from agent_env.store.document_store import Filter, UpdateSpec
from agent_env.task import Task, TaskLifecycleError, TaskPersistenceFailure
from agent_env.task import nested as nested_module
from agent_env.task import store as task_store
from agent_env.task.child_store import TASK_CHILD_INVOCATIONS_COLLECTION, ChildInvocation, get_child_invocation_store
from agent_env.task.nested import run_child
from agent_env.task.persistence import StoreDeadlineExceeded, current_store_deadline, store_deadline
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION, TASK_STEP_JOURNAL_COLLECTION, TaskStepStatus, get_task_instance_store,
)
from agent_env.task.teardown import TeardownReport
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep


CHILD_TIMEOUT_SECONDS = 5.0
CLEANUP_TIMEOUT_SECONDS = 5.0
PERSISTENCE_TIMEOUT_SECONDS = 0.05
TEST_DEADLINE_SECONDS = 10.0
BACKEND_RELEASE_SECONDS = 5.0
NAMED_CHILD_INVOCATIONS = ("artifact-review", "construction-review")
StoreFailureStage = Literal[
    "registration", "seed", "journal", "completion", "child_registration", "child_completion", "child_cleanup_start",
]


class PersistenceProbeStep(TaskStep):
    type = "required_persistence_probe"
    entity_refs = ()
    calls: list[str] = []

    @classmethod
    def from_dict(cls, data: dict) -> PersistenceProbeStep:
        return cls(**cls._base_from_dict(data))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        self.calls.append(self.id)
        context.metadata[self.id] = "published"
        return context


class ResourcePersistenceProbeStep(PersistenceProbeStep):
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        await super().execute(context)
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name="resource", sandbox_id="resource-id", sandbox_mode="container", sandbox_type="test-provider",
        ))
        return context


class PersistenceBudgetProbeStep(PersistenceProbeStep):
    type = "persistence_budget_probe"

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata[self.id] = current_store_deadline() - asyncio.get_running_loop().time()
        return context


class NamedChildrenProbeStep(PersistenceProbeStep):
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        for name in NAMED_CHILD_INVOCATIONS:
            child = Task(id=name, version=1, steps=[PersistenceProbeStep(id=name, version=None)])
            result = await run_child(
                step=self, context=context, task=child.to_dict(), inputs={}, iteration=0, invocation_name=name,
                child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS,
            )
            context.metadata[name] = result.metadata[name]
        return context


@pytest.fixture
def persistence_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalSqliteDocumentStore]:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[task_steps]\nimpls = ["tst.unit.task.required_persistence_test:PersistenceProbeStep", '
        '"tst.unit.task.required_persistence_test:PersistenceBudgetProbeStep"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config_path))
    documents = LocalSqliteDocumentStore(path=str(tmp_path / "documents.sqlite3"))
    configure(
        document_store=documents,
        object_store=LocalFilesystemObjectStore(root=str(tmp_path / "objects")),
    )
    monkeypatch.setattr(PersistenceProbeStep, "calls", [])
    yield documents
    reset_config()


def inject_store_failure(
    documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch, *,
    stage: StoreFailureStage, failure: RuntimeError,
    step_id: str = "write",
) -> None:
    insert = documents.insert
    update = documents.update_one_and_get

    def insert_document(collection: str, doc: dict) -> None:
        if collection == TASK_INSTANCES_COLLECTION and stage == "registration":
            raise failure
        if collection == TASK_CHILD_INVOCATIONS_COLLECTION and stage == "child_registration":
            raise failure
        if collection == TASK_STEP_JOURNAL_COLLECTION and stage == "journal" and doc.get("step_id") == step_id:
            raise failure
        insert(collection, doc)

    def update_document(
        collection: str, filter: Filter, changes: UpdateSpec, return_after: bool = True, upsert: bool = False,
    ) -> dict | None:
        if collection == TASK_INSTANCES_COLLECTION:
            if stage == "seed" and changes.set.get("context.metadata.initial") == "seed":
                raise failure
            if stage == "completion" and changes.set.get(f"context.metadata.{step_id}") == "published":
                raise failure
        if collection == TASK_CHILD_INVOCATIONS_COLLECTION:
            if stage == "child_completion" and changes.set.get("outcome") is not None:
                raise failure
            if stage == "child_cleanup_start" and changes.set.get("cleanup_state") == "running":
                raise failure
        return update(collection=collection, filter=filter, update=changes, return_after=return_after, upsert=upsert)

    monkeypatch.setattr(documents, "insert", insert_document)
    monkeypatch.setattr(documents, "update_one_and_get", update_document)


def build_probe_task(*, task_id: str = "required-persistence") -> Task:
    return Task(id=task_id, version=1, steps=[
        PersistenceProbeStep(
            id="write", version=None, fail_task_on_error=False,
            retry_config=RetryConfig(retry_from_step_id="write"),
        ),
        PersistenceProbeStep(id="dependent", version=None),
    ])


def build_nested_probe_task() -> Task:
    child = Task(id="required-child", version=1, steps=[PersistenceProbeStep(id="child_write", version=None)])
    return Task(id="required-parent", version=1, steps=[
        SubDAGTaskStep(
            id="child", version=None, task=child, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
            cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS, output_bindings={"output": "child_write"},
            fail_task_on_error=False, retry_config=RetryConfig(retry_from_step_id="child"),
        ),
        PersistenceProbeStep(id="dependent", version=None),
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["journal", "completion"])
async def test_required_commit_failure_never_publishes_outputs_unblocks_dependents_or_retries(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
    stage: Literal["journal", "completion"],
) -> None:
    original = RuntimeError("backend refused the write")
    inject_store_failure(persistence_documents, monkeypatch, stage=stage, failure=original)
    context = TaskStepContext()
    completed: list[str] = []

    with pytest.raises(TaskPersistenceFailure) as failure:
        await build_probe_task().run(
            context=context, require_persistence=True,
            on_step_complete=lambda index, total, step, snapshot, duration: completed.append(step.id),
        )

    assert isinstance(failure.value, TaskLifecycleError)
    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == ["write"]
    assert "write" not in context.metadata and "dependent" not in context.metadata
    assert completed == []
    instance = get_task_instance_store().get(context.instance_id)
    assert instance.status == "failed"
    assert all(step.status is TaskStepStatus.FAILURE for step in instance.completed_steps)
    assert "write" not in instance.context["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["journal", "completion"])
async def test_default_commit_policy_still_publishes_outputs_and_runs_dependents_after_store_error(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
    stage: Literal["journal", "completion"],
) -> None:
    inject_store_failure(persistence_documents, monkeypatch, stage=stage, failure=RuntimeError("backend refused the write"))

    result = await build_probe_task().run()

    assert PersistenceProbeStep.calls == ["write", "dependent"]
    assert result.metadata["write"] == "published" and result.metadata["dependent"] == "published"


@pytest.mark.asyncio
async def test_default_seed_policy_still_runs_after_store_error(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inject_store_failure(
        persistence_documents, monkeypatch, stage="seed", failure=RuntimeError("backend refused the write"),
    )

    result = await build_probe_task().run(context=TaskStepContext(metadata={"initial": "seed"}))

    assert PersistenceProbeStep.calls == ["write", "dependent"]
    assert result.metadata["write"] == "published" and result.metadata["dependent"] == "published"


@pytest.mark.asyncio
async def test_required_registration_failure_stops_before_any_step_and_does_not_leak_into_later_runs(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("instance insert failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="registration", failure=original)
    context = TaskStepContext()
    task = build_probe_task()

    with pytest.raises(TaskPersistenceFailure) as failure:
        await task.run(context=context, require_persistence=True)

    assert failure.value.__cause__ is original
    assert context.instance_id is None and PersistenceProbeStep.calls == []
    assert get_task_instance_store().find(task.id) == []

    result = await task.run()

    assert result.instance_id is None
    assert result.metadata["write"] == "published" and result.metadata["dependent"] == "published"


@pytest.mark.asyncio
async def test_required_seed_failure_preserves_cause_and_stops_before_any_step(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("seed update failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="seed", failure=original)
    context = TaskStepContext(metadata={"initial": "seed"})

    with pytest.raises(TaskPersistenceFailure) as failure:
        await build_probe_task().run(context=context, require_persistence=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == []
    instance = get_task_instance_store().get(context.instance_id)
    assert instance.status == "failed"
    assert instance.completed_steps == []


@pytest.mark.asyncio
async def test_cancellation_during_failure_write_retry_preserves_seed_error_and_terminal_status(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("seed update failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="seed", failure=original)
    update = persistence_documents.update
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    failed_once = False

    def update_document(collection, filter, changes):
        nonlocal failed_once
        if changes.set.get("status") == "failed" and not failed_once:
            failed_once = True
            loop.call_soon_threadsafe(entered.set)
            raise ConnectionError("temporary backend failure")
        return update(collection, filter, changes)

    monkeypatch.setattr(persistence_documents, "update", update_document)
    context = TaskStepContext(metadata={"initial": "seed"})
    running = asyncio.create_task(build_probe_task().run(context=context, require_persistence=True))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await entered.wait()
            running.cancel()
            with pytest.raises(TaskPersistenceFailure) as failure:
                await running
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == []
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_required_registration_rejects_a_missing_instance_instead_of_running_without_a_record(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_store, "register_task_instance", lambda *arguments, **keywords: None)
    context = TaskStepContext()

    with pytest.raises(TaskPersistenceFailure):
        await build_probe_task().run(context=context, require_persistence=True)

    assert context.instance_id is None and PersistenceProbeStep.calls == []


@pytest.mark.asyncio
async def test_required_resumed_registration_stops_when_a_retained_step_cannot_be_journaled(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("retained step journal failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="journal", failure=original)

    with pytest.raises(TaskPersistenceFailure) as failure:
        await build_probe_task().run(completed_step_ids={"write"}, require_persistence=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == []


@pytest.mark.asyncio
async def test_native_child_inherits_required_persistence_and_cannot_publish_a_failed_commit(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("child journal insert failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="journal", failure=original, step_id="child_write")
    task = build_nested_probe_task()
    context = TaskStepContext()

    with pytest.raises(TaskPersistenceFailure) as failure:
        await task.run(context=context, require_persistence=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == ["child_write"]
    assert "output" not in context.metadata and "dependent" not in context.metadata
    invocation, = get_child_invocation_store().list_for_parent(context.instance_id)
    assert invocation.outcome is None
    assert get_task_instance_store().get(invocation.child_instance_id).status == "failed"
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["child_registration", "child_completion", "child_cleanup_start"])
async def test_required_child_ledger_failure_is_fatal_even_for_a_tolerant_parent_step(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
    stage: StoreFailureStage,
) -> None:
    original = RuntimeError("child invocation write failed")
    inject_store_failure(persistence_documents, monkeypatch, stage=stage, failure=original)
    context = TaskStepContext()

    with pytest.raises(TaskPersistenceFailure) as failure:
        await build_nested_probe_task().run(context=context, require_persistence=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == ([] if stage == "child_registration" else ["child_write"])
    assert "output" not in context.metadata and "dependent" not in context.metadata
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_default_child_cleanup_policy_still_publishes_after_cleanup_start_could_not_be_recorded(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inject_store_failure(
        persistence_documents, monkeypatch, stage="child_cleanup_start", failure=RuntimeError("cleanup state write failed"),
    )

    result = await build_nested_probe_task().run()

    assert result.metadata["output"] == "published" and result.metadata["dependent"] == "published"
    assert PersistenceProbeStep.calls == ["child_write", "dependent"]


@pytest.mark.asyncio
async def test_parent_cancellation_retains_required_cleanup_recording_failure_after_teardown_finishes(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("cleanup state write failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="child_cleanup_start", failure=original)
    entered, released, finished, cancellation_dispatched = (asyncio.Event() for _ in range(4))

    async def controlled_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        entered.set()
        await released.wait()
        finished.set()
        return TeardownReport()

    monkeypatch.setattr(nested_module, "teardown_run", controlled_teardown)
    context = TaskStepContext()
    running = asyncio.create_task(build_nested_probe_task().run(context=context, require_persistence=True))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await entered.wait()
            running.cancel()
            asyncio.get_running_loop().call_soon(cancellation_dispatched.set)
            await cancellation_dispatched.wait()
            assert not running.done()
            released.set()
            with pytest.raises(TaskPersistenceFailure) as failure:
                await running
    finally:
        released.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert finished.is_set()
    causes = failure.value.__cause__
    assert isinstance(causes, BaseExceptionGroup)
    cancellation, persistence_failure = causes.exceptions
    assert isinstance(cancellation, asyncio.CancelledError)
    assert isinstance(persistence_failure, TaskPersistenceFailure) and persistence_failure.__cause__ is original
    assert "output" not in context.metadata and "dependent" not in context.metadata
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_parent_cancellation_during_final_cleanup_reporting_retains_the_required_recording_failure(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("cleanup state write failed")
    inject_store_failure(persistence_documents, monkeypatch, stage="child_cleanup_start", failure=original)
    entered, released, finished = threading.Event(), threading.Event(), threading.Event()
    store = get_child_invocation_store()
    capture = store.capture_cleanup_context
    captures = 0

    def capture_cleanup_context(invocation: ChildInvocation, *, context: dict) -> ChildInvocation:
        nonlocal captures
        captures += 1
        if captures != 2:
            return capture(invocation, context=context)
        entered.set()
        try:
            if not released.wait(BACKEND_RELEASE_SECONDS):
                raise AssertionError("the fixture did not release its blocked cleanup report")
            return capture(invocation, context=context)
        finally:
            finished.set()

    async def complete_teardown(context: TaskStepContext, *, timeout: float) -> TeardownReport:
        return TeardownReport()

    monkeypatch.setattr(store, "capture_cleanup_context", capture_cleanup_context)
    monkeypatch.setattr(nested_module, "teardown_run", complete_teardown)
    context = TaskStepContext()
    running = asyncio.create_task(build_nested_probe_task().run(context=context, require_persistence=True))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
            running.cancel()
            with pytest.raises(TaskPersistenceFailure) as failure:
                await running
    finally:
        released.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)

    causes = failure.value.__cause__
    assert isinstance(causes, BaseExceptionGroup)
    cancellation, persistence_failure = causes.exceptions
    assert isinstance(cancellation, asyncio.CancelledError)
    assert isinstance(persistence_failure, TaskPersistenceFailure) and persistence_failure.__cause__ is original
    assert "output" not in context.metadata and "dependent" not in context.metadata
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_required_resource_checkpoint_failure_retains_cleanup_ownership_without_publishing_step_outputs(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("resource ownership write failed")
    update = persistence_documents.update_one_and_get

    def update_document(
        collection: str, filter: Filter, changes: UpdateSpec, return_after: bool = True, upsert: bool = False,
    ) -> dict | None:
        if collection == TASK_INSTANCES_COLLECTION and changes.add_to_set.get("resource_context.deployed_sandboxes"):
            raise original
        return update(collection=collection, filter=filter, update=changes, return_after=return_after, upsert=upsert)

    monkeypatch.setattr(persistence_documents, "update_one_and_get", update_document)
    context = TaskStepContext()
    task = Task(id="required-resource", version=1, steps=[
        ResourcePersistenceProbeStep(id="resource_write", version=None, fail_task_on_error=False),
        PersistenceProbeStep(id="dependent", version=None),
    ])

    with pytest.raises(TaskPersistenceFailure) as failure:
        await task.run(context=context, require_persistence=True)

    assert failure.value.__cause__ is original
    assert PersistenceProbeStep.calls == ["resource_write"]
    assert "resource_write" not in context.metadata and "dependent" not in context.metadata
    assert [sandbox.sandbox_id for sandbox in context.deployed_sandboxes] == ["resource-id"]
    assert get_task_instance_store().get(context.instance_id).status == "failed"


@pytest.mark.asyncio
async def test_named_child_graphs_from_one_owner_resume_without_colliding_or_replaying_their_work(
    persistence_documents: LocalSqliteDocumentStore,
) -> None:
    task = Task(id="named-child-parent", version=1, steps=[NamedChildrenProbeStep(id="dispatch", version=None)])

    first = await task.run(instance_id="named-parent-run", require_persistence=True)
    resumed = await task.run(instance_id="named-parent-run", require_persistence=True)

    assert [first.metadata[name] for name in NAMED_CHILD_INVOCATIONS] == ["published"] * len(NAMED_CHILD_INVOCATIONS)
    assert [resumed.metadata[name] for name in NAMED_CHILD_INVOCATIONS] == [
        first.metadata[name] for name in NAMED_CHILD_INVOCATIONS
    ]
    assert PersistenceProbeStep.calls == list(NAMED_CHILD_INVOCATIONS)
    children = get_child_invocation_store().list_for_parent("named-parent-run")
    assert [child.invocation_name for child in children] == list(NAMED_CHILD_INVOCATIONS)
    assert all(child.parent_step_id == "dispatch" and child.iteration == 0 for child in children)


@pytest.mark.asyncio
async def test_task_persistence_budget_is_inherited_by_native_children(
    persistence_documents: LocalSqliteDocumentStore,
) -> None:
    child = Task(id="budget-child", version=1, steps=[PersistenceBudgetProbeStep(id="child_budget", version=None)])
    task = Task(id="budget-parent", version=1, steps=[
        PersistenceBudgetProbeStep(id="parent_budget", version=None),
        SubDAGTaskStep(
            id="child", version=None, task=child, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
            cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS, output_bindings={"child_budget": "child_budget"},
        ),
    ])

    result = await task.run(require_persistence=True, persistence_timeout_seconds=PERSISTENCE_TIMEOUT_SECONDS)

    assert 0 < result.metadata["parent_budget"] <= PERSISTENCE_TIMEOUT_SECONDS
    assert 0 < result.metadata["child_budget"] <= PERSISTENCE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_task_persistence_budget_never_extends_an_inherited_execution_deadline(
    persistence_documents: LocalSqliteDocumentStore,
) -> None:
    task = Task(id="budget-with-deadline", version=1, steps=[PersistenceBudgetProbeStep(id="budget", version=None)])
    deadline = asyncio.get_running_loop().time() + CHILD_TIMEOUT_SECONDS

    with store_deadline(deadline):
        result = await task.run(require_persistence=True, persistence_timeout_seconds=TEST_DEADLINE_SECONDS)

    assert 0 < result.metadata["budget"] <= CHILD_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_task_persistence_budget_bounds_a_blocked_registration_without_starting_step_work(
    persistence_documents: LocalSqliteDocumentStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, released, finished = threading.Event(), threading.Event(), threading.Event()
    insert = persistence_documents.insert

    def insert_document(collection: str, document: dict) -> None:
        if collection != TASK_INSTANCES_COLLECTION:
            insert(collection, document)
            return
        entered.set()
        try:
            if not released.wait(BACKEND_RELEASE_SECONDS):
                raise AssertionError("the fixture did not release the blocked backend")
            raise RuntimeError("the backend completed after its caller left")
        finally:
            finished.set()

    monkeypatch.setattr(persistence_documents, "insert", insert_document)
    pending = asyncio.create_task(build_probe_task().run(
        require_persistence=True, persistence_timeout_seconds=PERSISTENCE_TIMEOUT_SECONDS,
    ))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
            with pytest.raises(StoreDeadlineExceeded):
                await pending
            assert PersistenceProbeStep.calls == []
    finally:
        released.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0, -1, float("nan"), float("inf"), True])
async def test_task_persistence_budget_rejects_nonpositive_nonfinite_or_boolean_configuration(
    persistence_documents: LocalSqliteDocumentStore, timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="persistence_timeout_seconds"):
        await build_probe_task().run(persistence_timeout_seconds=timeout_seconds)
    assert PersistenceProbeStep.calls == []
