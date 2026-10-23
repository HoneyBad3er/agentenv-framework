"""Durable child identities outlive parent step heartbeats and keep resumed invocations pinned."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from agent_env.config import get_config
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import DocumentStore, DuplicateKeyError, Filter, Sort, SortKey, UpdateSpec, compare_and_swap
from agent_env.task.store import TASK_INSTANCES_COLLECTION, SupersededAttemptError, TaskInstance
from agent_env.task_step.context import TaskStepContext

TASK_CHILD_INVOCATIONS_COLLECTION = "task_child_invocations"
_KEY_FIELDS = ["parent_instance_id", "parent_step_id", "parent_step_generation", "iteration"]


class ChildInvocationConflict(ValueError):
    """A logical invocation already pins a different definition, inputs or completed outcome."""


class ChildCleanupState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _json_snapshot(value: dict) -> dict:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    snapshot = json.loads(encoded)
    if not isinstance(snapshot, dict) or snapshot != value:
        raise ValueError("child invocation values must be JSON objects")
    return snapshot


def _same_json(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ChildInvocation:
    parent_instance_id: str
    parent_step_id: str
    parent_step_generation: int
    parent_attempt_epoch: int
    iteration: int
    child_instance_id: str
    task_definition: dict
    captured_inputs: dict
    outcome: dict | None = None
    cleanup_state: ChildCleanupState = ChildCleanupState.PENDING
    cleanup_error: str | None = None
    rev: int = 0

    def to_dict(self) -> dict:
        return copy.deepcopy({
            "parent_instance_id": self.parent_instance_id, "parent_step_id": self.parent_step_id,
            "parent_step_generation": self.parent_step_generation, "parent_attempt_epoch": self.parent_attempt_epoch,
            "iteration": self.iteration, "child_instance_id": self.child_instance_id,
            "task_definition": self.task_definition, "captured_inputs": self.captured_inputs,
            "outcome": self.outcome, "cleanup_state": str(self.cleanup_state), "cleanup_error": self.cleanup_error,
            "rev": self.rev,
        })

    @classmethod
    def from_dict(cls, document: dict) -> ChildInvocation:
        return cls(
            parent_instance_id=document["parent_instance_id"], parent_step_id=document["parent_step_id"],
            parent_step_generation=document["parent_step_generation"], parent_attempt_epoch=document["parent_attempt_epoch"],
            iteration=document["iteration"], child_instance_id=document["child_instance_id"],
            task_definition=copy.deepcopy(document["task_definition"]), captured_inputs=copy.deepcopy(document["captured_inputs"]),
            outcome=copy.deepcopy(document.get("outcome")), cleanup_state=ChildCleanupState(document["cleanup_state"]),
            cleanup_error=document.get("cleanup_error"), rev=document.get("rev", 0),
        )


class ChildInvocationStore:
    def __init__(self) -> None:
        self._indexed: DocumentStore | None = None

    @property
    def _doc_store(self) -> DocumentStore:
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(TASK_CHILD_INVOCATIONS_COLLECTION, _KEY_FIELDS, unique=True)
            store.ensure_index(TASK_CHILD_INVOCATIONS_COLLECTION, ["child_instance_id"], unique=True)
            store.ensure_index(TASK_CHILD_INVOCATIONS_COLLECTION, ["parent_instance_id"])
            self._indexed = store
        return store

    def _check_parent(self, store: DocumentStore, invocation: ChildInvocation) -> None:
        document = store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=invocation.parent_instance_id))
        if document is None:
            raise NotFoundError(f"TaskInstance {invocation.parent_instance_id!r} not found")
        parent = TaskInstance.from_dict(document)
        if parent.attempt_epoch != invocation.parent_attempt_epoch:
            raise SupersededAttemptError(parent.instance_id, invocation.parent_step_id,
                                         "another attempt owns the parent instance")
        if parent.child_step_generations.get(invocation.parent_step_id, 0) != invocation.parent_step_generation:
            raise SupersededAttemptError(parent.instance_id, invocation.parent_step_id,
                                         "the parent step was rolled back")

    def _key(self, invocation: ChildInvocation) -> Filter:
        return Filter.of(parent_instance_id=invocation.parent_instance_id, parent_step_id=invocation.parent_step_id,
                         parent_step_generation=invocation.parent_step_generation, iteration=invocation.iteration)

    def _check_identity(self, current: ChildInvocation, expected: ChildInvocation) -> None:
        if (not _same_json(current.task_definition, expected.task_definition)
                or not _same_json(current.captured_inputs, expected.captured_inputs)):
            raise ChildInvocationConflict("a child invocation already pins different task sources or captured inputs")

    def get_or_create(
        self, *, parent_instance_id: str, parent_step_id: str, parent_step_generation: int,
        parent_attempt_epoch: int, iteration: int, task_definition: dict, captured_inputs: dict,
    ) -> ChildInvocation:
        candidate = ChildInvocation(
            parent_instance_id=parent_instance_id, parent_step_id=parent_step_id,
            parent_step_generation=parent_step_generation, parent_attempt_epoch=parent_attempt_epoch, iteration=iteration,
            child_instance_id=f"{parent_instance_id}/child-{uuid.uuid4().hex}", task_definition=_json_snapshot(task_definition),
            captured_inputs=_json_snapshot(captured_inputs),
        )
        store = self._doc_store
        self._check_parent(store, candidate)
        try:
            store.insert(TASK_CHILD_INVOCATIONS_COLLECTION, candidate.to_dict())
        except DuplicateKeyError:
            def adopt(document: dict) -> UpdateSpec:
                self._check_parent(store, candidate)
                current = ChildInvocation.from_dict(document)
                self._check_identity(current, candidate)
                if current.parent_attempt_epoch > parent_attempt_epoch:
                    raise SupersededAttemptError(parent_instance_id, parent_step_id, "another attempt owns the child invocation")
                return UpdateSpec(set={"parent_attempt_epoch": parent_attempt_epoch})

            document = compare_and_swap(store, TASK_CHILD_INVOCATIONS_COLLECTION, self._key(candidate),
                                        adopt, counter_field="rev")
            if document is None:
                raise NotFoundError("the child invocation disappeared while acquiring its recorded identity")
            candidate = ChildInvocation.from_dict(document)
        self._check_parent(store, candidate)
        return candidate

    def get(
        self, *, parent_instance_id: str, parent_step_id: str, parent_step_generation: int,
        parent_attempt_epoch: int, iteration: int,
    ) -> ChildInvocation | None:
        expected = ChildInvocation(parent_instance_id=parent_instance_id, parent_step_id=parent_step_id,
                                   parent_step_generation=parent_step_generation, parent_attempt_epoch=parent_attempt_epoch,
                                   iteration=iteration, child_instance_id="", task_definition={}, captured_inputs={})
        store = self._doc_store
        self._check_parent(store, expected)
        document = store.find_one(TASK_CHILD_INVOCATIONS_COLLECTION, self._key(expected))
        self._check_parent(store, expected)
        return ChildInvocation.from_dict(document) if document is not None else None

    def list_for_parent(self, parent_instance_id: str, *, parent_attempt_epoch: int | None = None) -> list[ChildInvocation]:
        match = Filter.of(parent_instance_id=parent_instance_id)
        if parent_attempt_epoch is not None:
            match = Filter.of(parent_instance_id=parent_instance_id, parent_attempt_epoch=parent_attempt_epoch)
        documents = self._doc_store.query(
            TASK_CHILD_INVOCATIONS_COLLECTION, match,
            sort=Sort((SortKey("parent_step_id", descending=False), SortKey("parent_step_generation", descending=False),
                       SortKey("iteration", descending=False))),
        )
        return [ChildInvocation.from_dict(document) for document in documents]

    def _update(self, invocation: ChildInvocation, mutate: Callable[[ChildInvocation], UpdateSpec]) -> ChildInvocation:
        store = self._doc_store
        self._check_parent(store, invocation)

        def update(document: dict) -> UpdateSpec:
            self._check_parent(store, invocation)
            current = ChildInvocation.from_dict(document)
            if (current.parent_attempt_epoch != invocation.parent_attempt_epoch
                    or current.child_instance_id != invocation.child_instance_id):
                raise SupersededAttemptError(invocation.parent_instance_id, invocation.parent_step_id,
                                             "another attempt owns the child invocation")
            return mutate(current)

        document = compare_and_swap(store, TASK_CHILD_INVOCATIONS_COLLECTION, self._key(invocation), update, counter_field="rev")
        if document is None:
            raise NotFoundError("the child invocation disappeared before its outcome was recorded")
        self._check_parent(store, invocation)
        return ChildInvocation.from_dict(document)

    def complete(self, invocation: ChildInvocation, *, outcome: dict) -> ChildInvocation:
        captured = _json_snapshot(TaskStepContext.from_dict(outcome).to_safe_dict())

        def finish(current: ChildInvocation) -> UpdateSpec:
            if current.outcome is not None and not _same_json(current.outcome, captured):
                raise ChildInvocationConflict("a completed child invocation already pins a different outcome")
            return UpdateSpec(set={"outcome": captured})

        return self._update(invocation, finish)

    def set_cleanup_state(
        self, invocation: ChildInvocation, *, state: ChildCleanupState, error: str | None = None,
    ) -> ChildInvocation:
        state = ChildCleanupState(state)

        def record(current: ChildInvocation) -> UpdateSpec:
            if current.cleanup_state is ChildCleanupState.COMPLETED and state is not ChildCleanupState.COMPLETED:
                if current.outcome is not None or state is not ChildCleanupState.PENDING:
                    raise ChildInvocationConflict("completed child cleanup cannot be reopened")
            return UpdateSpec(set={"cleanup_state": str(state), "cleanup_error": error})

        return self._update(invocation, record)


_child_invocation_store: ChildInvocationStore | None = None


def get_child_invocation_store() -> ChildInvocationStore:
    global _child_invocation_store
    if _child_invocation_store is None:
        _child_invocation_store = ChildInvocationStore()
    return _child_invocation_store


def reset_child_invocation_store() -> None:
    global _child_invocation_store
    _child_invocation_store = None
