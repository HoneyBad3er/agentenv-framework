"""Child identity and completion survive a parent resume while rollback retires the affected logical calls."""

from __future__ import annotations

import concurrent.futures
import threading
import uuid
from dataclasses import dataclass

import pytest

from agent_env.config import configure, set_document_store
from agent_env.store import Filter, LocalSqliteDocumentStore, UpdateSpec
from agent_env.store.base import ConcurrentModificationError
from agent_env.store.routing import run_scope
from agent_env.task import Task
from agent_env.task.child_store import (
    TASK_CHILD_INVOCATIONS_COLLECTION, ChildCleanupState, ChildInvocationConflict, ChildInvocationStore,
)
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION, TASK_STEP_JOURNAL_COLLECTION, SupersededAttemptError, TaskInstance, TaskInstanceStore,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps

CONTENDING_WRITERS = 8
THREAD_DEADLINE_SECONDS = 15


@dataclass(frozen=True)
class Stores:
    documents: LocalSqliteDocumentStore
    instances: TaskInstanceStore
    parent: TaskInstance
    children: ChildInvocationStore


@pytest.fixture
def stores(tmp_path) -> Stores:
    documents = LocalSqliteDocumentStore(str(tmp_path / "children.sqlite3"))
    set_document_store(documents)
    instances = TaskInstanceStore()
    parent = instances.create_instance("parent", 1, total_steps=3)
    instances.seed_context(parent.instance_id, ContextUpdateOps())
    return Stores(documents, instances, parent, ChildInvocationStore())


def create_child(
    stores: Stores, *, epoch: int = 0, generation: int = 0, iteration: int = 0, step_id: str = "loop",
    task_definition: dict | None = None, captured_inputs: dict | None = None,
):
    return stores.children.get_or_create(
        parent_instance_id=stores.parent.instance_id, parent_step_id=step_id,
        parent_step_generation=generation, parent_attempt_epoch=epoch, iteration=iteration,
        task_definition={"id": "child", "version": 7, "type": "task", "steps": []}
        if task_definition is None else task_definition,
        captured_inputs={"candidate": "captured"} if captured_inputs is None else captured_inputs,
    )


def test_competing_creation_persists_one_child_uuid_before_any_child_execution(stores):
    ready = threading.Barrier(CONTENDING_WRITERS)

    def create():
        ready.wait(timeout=THREAD_DEADLINE_SECONDS)
        return create_child(stores)

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONTENDING_WRITERS) as executor:
        pending = [executor.submit(create) for _ in range(CONTENDING_WRITERS)]
        invocations = [future.result(timeout=THREAD_DEADLINE_SECONDS) for future in pending]

    assert len({invocation.child_instance_id for invocation in invocations}) == 1
    assert invocations[0].child_instance_id.startswith(stores.parent.instance_id + "/child-")
    assert uuid.UUID(invocations[0].child_instance_id.rsplit("/child-", 1)[1])
    assert stores.documents.count(TASK_CHILD_INVOCATIONS_COLLECTION, Filter()) == 1
    assert stores.instances.get(stores.parent.instance_id).completed_steps == []


def test_new_store_reloads_completed_child_independently_of_parent_step_completion(stores):
    child = create_child(stores)
    outcome = TaskStepContext(metadata={"answer": 42}, instance_id=child.child_instance_id)
    completed = stores.children.complete(child, outcome=outcome.to_safe_dict())
    stores.children.set_cleanup_state(completed, state=ChildCleanupState.COMPLETED)

    reopened = ChildInvocationStore().get(
        parent_instance_id=stores.parent.instance_id, parent_step_id="loop", parent_step_generation=0,
        parent_attempt_epoch=0, iteration=0,
    )

    assert reopened.child_instance_id == child.child_instance_id
    assert reopened.outcome == outcome.to_safe_dict()
    assert reopened.cleanup_state is ChildCleanupState.COMPLETED
    assert stores.instances.get(stores.parent.instance_id).completed_steps == []


@pytest.mark.parametrize("changed", ["task_definition", "captured_inputs"])
def test_logical_call_rejects_changed_task_sources_or_captured_inputs(stores, changed):
    child = create_child(stores)
    options = {"task_definition": child.task_definition, "captured_inputs": child.captured_inputs}
    options[changed] = {"different": True}

    with pytest.raises(ChildInvocationConflict):
        create_child(stores, **options)

    assert stores.children.list_for_parent(stores.parent.instance_id)[0].child_instance_id == child.child_instance_id


def test_resume_adopts_same_logical_child_and_fences_previous_attempt_writers(stores):
    child = create_child(stores)
    completed = stores.children.complete(child, outcome=TaskStepContext(metadata={"answer": 42}).to_safe_dict())
    parent = stores.instances.upsert_instance(stores.parent.instance_id, "parent", 1, total_steps=3)

    adopted = create_child(stores, epoch=parent.attempt_epoch)

    assert adopted.child_instance_id == child.child_instance_id
    assert adopted.outcome == completed.outcome
    assert adopted.parent_attempt_epoch == parent.attempt_epoch
    with pytest.raises(SupersededAttemptError):
        stores.children.complete(child, outcome=TaskStepContext(metadata={"answer": 99}).to_safe_dict())
    with pytest.raises(SupersededAttemptError):
        stores.children.set_cleanup_state(child, state=ChildCleanupState.COMPLETED)
    stores.children.set_cleanup_state(adopted, state=ChildCleanupState.COMPLETED)
    assert stores.children.list_for_parent(stores.parent.instance_id, parent_attempt_epoch=0) == []


def test_rollback_retires_only_named_steps_children_and_resume_preserves_generations(stores):
    retired = create_child(stores)
    retained = create_child(stores, step_id="sibling")

    undone = stores.instances.undo_steps_sync(stores.parent.instance_id, {"loop"}, bump_epoch=True, attempt_epoch=0)
    parent = stores.instances.upsert_instance(stores.parent.instance_id, "parent", 1, total_steps=3)

    assert undone["child_step_generations"] == {"loop": 1}
    assert parent.child_step_generations == {"loop": 1}
    replacement = create_child(stores, epoch=parent.attempt_epoch, generation=1)
    preserved = create_child(stores, epoch=parent.attempt_epoch, step_id="sibling")
    assert replacement.child_instance_id != retired.child_instance_id
    assert preserved.child_instance_id == retained.child_instance_id
    with pytest.raises(SupersededAttemptError):
        create_child(stores, epoch=parent.attempt_epoch, generation=0)
    assert len(stores.children.list_for_parent(stores.parent.instance_id)) == 3


def test_rollback_increments_generation_even_for_an_inflight_uncompleted_step(stores):
    stores.instances.undo_steps_sync(stores.parent.instance_id, {"loop"})
    stores.instances.undo_steps_sync(stores.parent.instance_id, {"loop"})

    parent = stores.instances.get(stores.parent.instance_id)

    assert parent.child_step_generations == {"loop": 2}
    assert parent.attempt_epoch == 0


def test_completed_outcome_and_finished_cleanup_cannot_be_overwritten(stores):
    child = create_child(stores)
    completed = stores.children.complete(child, outcome=TaskStepContext(metadata={"answer": 1}).to_safe_dict())

    with pytest.raises(ChildInvocationConflict):
        stores.children.complete(completed, outcome=TaskStepContext(metadata={"answer": 2}).to_safe_dict())
    cleaned = stores.children.set_cleanup_state(completed, state=ChildCleanupState.COMPLETED)
    with pytest.raises(ChildInvocationConflict):
        stores.children.set_cleanup_state(cleaned, state=ChildCleanupState.RUNNING)


def test_resume_can_restart_cleaned_unfinished_child_but_not_a_durably_completed_one(stores):
    child = create_child(stores)
    stores.children.set_cleanup_state(child, state=ChildCleanupState.COMPLETED)
    parent = stores.instances.upsert_instance(stores.parent.instance_id, "parent", 1, total_steps=3)
    adopted = create_child(stores, epoch=parent.attempt_epoch)

    restarted = stores.children.set_cleanup_state(adopted, state=ChildCleanupState.PENDING)

    assert restarted.child_instance_id == child.child_instance_id
    assert restarted.outcome is None
    assert restarted.cleanup_state is ChildCleanupState.PENDING
    completed = stores.children.complete(restarted, outcome=TaskStepContext(metadata={"answer": 42}).to_safe_dict())
    cleaned = stores.children.set_cleanup_state(completed, state=ChildCleanupState.COMPLETED)
    with pytest.raises(ChildInvocationConflict):
        stores.children.set_cleanup_state(cleaned, state=ChildCleanupState.PENDING)


def test_completed_context_redacts_secrets_and_keeps_secret_references(stores):
    child = create_child(stores)
    context = TaskStepContext(metadata={
        "litellm_api_key": "fixture-secret", "_heartbeat_fn": lambda: None,
        "nested": {"judge_litellm_api_key": "secret:judge", "usersim_api_key": "env:SIM?fixture-secret"},
    })

    completed = stores.children.complete(child, outcome=context.to_dict())

    assert completed.outcome["metadata"] == {"nested": {"judge_litellm_api_key": "secret:judge"}}


def test_parent_epoch_change_during_insert_is_reported_and_the_link_remains_recoverable(stores, monkeypatch):
    insert = stores.documents.insert

    def insert_then_resume(collection, document):
        insert(collection, document)
        if collection == TASK_CHILD_INVOCATIONS_COLLECTION:
            stores.documents.update(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=stores.parent.instance_id),
                                    UpdateSpec(inc={"attempt_epoch": 1}))

    monkeypatch.setattr(stores.documents, "insert", insert_then_resume)

    with pytest.raises(SupersededAttemptError):
        create_child(stores)
    monkeypatch.setattr(stores.documents, "insert", insert)
    previous = stores.children.list_for_parent(stores.parent.instance_id)[0]
    recovered = create_child(stores, epoch=1)
    assert recovered.child_instance_id == previous.child_instance_id


def test_input_snapshot_is_independent_of_mutations_after_creation(stores):
    definition = {"id": "child", "version": 7, "steps": [{"id": "work"}]}
    inputs = {"items": [1]}
    child = create_child(stores, task_definition=definition, captured_inputs=inputs)
    definition["steps"].append({"id": "changed"})
    inputs["items"].append(2)

    assert child.task_definition["steps"] == [{"id": "work"}]
    assert child.captured_inputs == {"items": [1]}


class RefusingRecordsStore(LocalSqliteDocumentStore):
    def insert(self, collection, document):
        raise AssertionError(f"configured records written: {collection}")

    def update(self, collection, filter, update, upsert=False):
        raise AssertionError(f"configured records updated: {collection}")

    def update_one_and_get(self, collection, filter, update, return_after=True, upsert=False):
        raise AssertionError(f"configured records updated: {collection}")

    def find_one(self, collection, filter, sort=None):
        raise AssertionError(f"configured records read: {collection}")

    def query(self, collection, filter, sort=None, limit=None, offset=0):
        raise AssertionError(f"configured records queried: {collection}")


def test_local_child_identity_and_ledger_remain_local_outside_the_parent_run_scope(tmp_path, cli_routing):
    configured = RefusingRecordsStore(str(tmp_path / "configured.sqlite3"))
    configure(document_store=configured)
    instances = TaskInstanceStore()
    children = ChildInvocationStore()
    with run_scope("@local/~/fixture/tasks/parent"):
        parent = instances.create_instance("@local/~/fixture/tasks/parent", 1, total_steps=1)
        invocation = children.get_or_create(
            parent_instance_id=parent.instance_id, parent_step_id="nested", parent_step_generation=0,
            parent_attempt_epoch=0, iteration=0, task_definition={"id": "registry-child", "version": 1, "steps": []},
            captured_inputs={},
        )

    child = instances.upsert_instance(invocation.child_instance_id, "registry-child", 1, total_steps=1)
    assert child.instance_id.startswith("@local/")
    assert Task.get_instance(child.instance_id).task_id == "registry-child"
    completed = children.complete(invocation, outcome=TaskStepContext(metadata={"answer": 42}).to_safe_dict())
    cleaned = children.set_cleanup_state(completed, state=ChildCleanupState.COMPLETED)
    assert children.list_for_parent(parent.instance_id) == [cleaned]
    loaded = children.get(parent_instance_id=parent.instance_id, parent_step_id="nested", parent_step_generation=0,
                          parent_attempt_epoch=0, iteration=0)
    assert loaded.outcome["metadata"] == {"answer": 42}


def test_resume_replay_keeps_independent_successes_and_excludes_failed_partial_writes_without_retiring_children(stores):
    child = create_child(stores)
    for step_id in ("left", "independent"):
        stores.instances.record_step_complete_sync(
            stores.parent.instance_id, {"step_id": step_id, "status": "success"},
            ContextUpdateOps(sets={f"context.metadata.{step_id}": "committed"}), 3, "now",
        )
    stores.instances.journal_step_sync(stores.parent.instance_id, "uncommitted",
                                      ContextUpdateOps(sets={"context.metadata.uncommitted": "not retained"}), "success")
    dirty = TaskStepContext(metadata={"left": "committed", "independent": "committed", "partial": "not retained"})
    stores.instances.record_task_failure_sync(stores.parent.instance_id, "failed", dirty, RuntimeError("fixture failure"), "now")
    before = stores.instances.get(stores.parent.instance_id)

    replayed = stores.instances.replay_completed_context_sync(stores.parent.instance_id, {"left", "independent"})

    assert replayed["metadata"] == {"left": "committed", "independent": "committed"}
    assert create_child(stores).child_instance_id == child.child_instance_id
    after = stores.instances.get(stores.parent.instance_id)
    assert (after.rev, after.attempt_epoch, after.child_step_generations) == (
        before.rev, before.attempt_epoch, before.child_step_generations,
    )


@pytest.mark.parametrize("missing", ["seed", "step"])
def test_resume_replay_rejects_missing_journal_evidence(stores, missing):
    stores.instances.record_step_complete_sync(
        stores.parent.instance_id, {"step_id": "completed", "status": "success"}, ContextUpdateOps(), 3, "now",
    )
    stores.documents.delete(TASK_STEP_JOURNAL_COLLECTION,
                            Filter.of(instance_id=stores.parent.instance_id, step_id="__seed__" if missing == "seed" else "completed"))

    with pytest.raises(ValueError):
        stores.instances.replay_completed_context_sync(stores.parent.instance_id, {"completed"})


@pytest.mark.parametrize("field", ["rev", "attempt_epoch", "run_generation", "journal_seq"])
def test_resume_replay_refuses_instance_changes_while_reading_journals(stores, monkeypatch, field):
    query = stores.documents.query

    def query_then_advance(collection, filter, **kwargs):
        documents = query(collection, filter, **kwargs)
        if collection == TASK_STEP_JOURNAL_COLLECTION:
            stores.documents.update(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=stores.parent.instance_id),
                                    UpdateSpec(inc={field: 1}))
        return documents

    monkeypatch.setattr(stores.documents, "query", query_then_advance)

    with pytest.raises(ConcurrentModificationError):
        stores.instances.replay_completed_context_sync(stores.parent.instance_id, set())


def test_resume_replay_rejects_a_failed_step_as_a_retained_completion(stores):
    stores.instances.record_task_failure_sync(stores.parent.instance_id, "failed", TaskStepContext(),
                                             RuntimeError("fixture failure"), "now")

    with pytest.raises(ValueError):
        stores.instances.replay_completed_context_sync(stores.parent.instance_id, {"failed"})
