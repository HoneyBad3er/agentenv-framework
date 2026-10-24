"""A slow synchronous store cannot block task deadlines or grow an unbounded queue of abandoned writes."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from functools import partial

import pytest

from agent_env.config import set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.store.document_store import Filter, UpdateSpec
from agent_env.store.routing import in_local_run, run_scope
from agent_env.task.execution import TaskLifecycleError
from agent_env.task.persistence import (
    MAX_CONCURRENT_STORE_CALLS, StoreCallAbandoned, StoreDeadlineExceeded, call_store, current_store_deadline, store_deadline,
)
from agent_env.task.store import TASK_INSTANCES_COLLECTION, SupersededAttemptError, TaskInstanceStore, register_task_instance
from agent_env.task_step.context_ops import ContextUpdateOps

CALL_DEADLINE_SECONDS = 0.05
TEST_DEADLINE_SECONDS = 10.0
BACKEND_RELEASE_SECONDS = 5.0


def wait_for_release(event: threading.Event) -> None:
    if not event.wait(BACKEND_RELEASE_SECONDS):
        raise AssertionError("the test did not release its blocked backend")


@pytest.mark.asyncio
async def test_slow_store_returns_at_deadline_while_the_event_loop_remains_responsive() -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_store() -> None:
        entered.set()
        try:
            wait_for_release(release)
        finally:
            finished.set()

    loop = asyncio.get_running_loop()
    pending = asyncio.create_task(call_store(slow_store, deadline=loop.time() + CALL_DEADLINE_SECONDS))
    heartbeat = asyncio.Event()
    loop.call_soon(heartbeat.set)
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await heartbeat.wait()
            assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
            with pytest.raises(StoreDeadlineExceeded) as expired:
                await pending
            assert isinstance(expired.value, TaskLifecycleError)
            assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_timed_out_backend_calls_keep_capacity_until_their_threads_really_finish() -> None:
    entered = threading.Barrier(MAX_CONCURRENT_STORE_CALLS + 1)
    release, finished, overflow_started = threading.Event(), threading.Event(), threading.Event()
    completed = 0
    guard = threading.Lock()

    def slow_store() -> None:
        nonlocal completed
        entered.wait(timeout=BACKEND_RELEASE_SECONDS)
        try:
            wait_for_release(release)
        finally:
            with guard:
                completed += 1
                if completed == MAX_CONCURRENT_STORE_CALLS:
                    finished.set()

    loop = asyncio.get_running_loop()
    pending = [asyncio.create_task(call_store(slow_store, deadline=loop.time() + CALL_DEADLINE_SECONDS))
               for _ in range(MAX_CONCURRENT_STORE_CALLS)]
    try:
        await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
        outcomes = await asyncio.gather(*pending, return_exceptions=True)
        assert all(isinstance(outcome, StoreDeadlineExceeded) for outcome in outcomes)
        with pytest.raises(StoreDeadlineExceeded):
            await call_store(overflow_started.set, deadline=loop.time() + CALL_DEADLINE_SECONDS)
        assert not overflow_started.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_store_offload_preserves_namespace_and_execution_deadline_contextvars(cli_routing) -> None:
    identity: contextvars.ContextVar[str] = contextvars.ContextVar("fixture_identity", default="missing")
    token = identity.set("fixture-caller")
    deadline = asyncio.get_running_loop().time() + TEST_DEADLINE_SECONDS
    try:
        with run_scope("@local/~/fixture/tasks/parent"), store_deadline(deadline):
            result = await call_store(lambda: (identity.get(), in_local_run(), current_store_deadline()), deadline=deadline)
        assert result == ("fixture-caller", True, deadline)
    finally:
        identity.reset(token)


@pytest.mark.asyncio
async def test_cleanup_can_install_a_fresh_store_deadline_and_restore_the_execution_deadline() -> None:
    loop = asyncio.get_running_loop()
    execution_deadline = loop.time() - CALL_DEADLINE_SECONDS
    cleanup_deadline = loop.time() + TEST_DEADLINE_SECONDS
    with store_deadline(execution_deadline):
        with store_deadline(cleanup_deadline):
            assert current_store_deadline() == cleanup_deadline
            assert await call_store(lambda: "cleaned", deadline=current_store_deadline()) == "cleaned"
        assert current_store_deadline() == execution_deadline
    assert current_store_deadline() > loop.time()


@pytest.mark.asyncio
async def test_expired_store_deadline_does_not_start_backend_work() -> None:
    started = threading.Event()
    with pytest.raises(StoreDeadlineExceeded):
        await call_store(started.set, deadline=asyncio.get_running_loop().time() - CALL_DEADLINE_SECONDS)
    assert not started.is_set()


@pytest.mark.asyncio
async def test_backend_timeout_is_not_misreported_as_the_callers_persistence_deadline() -> None:
    failure = TimeoutError("backend timeout")

    def fail() -> None:
        raise failure

    with pytest.raises(TimeoutError) as expired:
        await call_store(fail, deadline=asyncio.get_running_loop().time() + TEST_DEADLINE_SECONDS)
    assert expired.value is failure
    assert not isinstance(expired.value, StoreDeadlineExceeded)


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_stage", ["context", "journal"])
async def test_seed_write_finishing_after_timeout_cannot_replace_a_new_attempts_context_or_journal(
    tmp_path, monkeypatch, delayed_stage: str,
) -> None:
    documents = LocalSqliteDocumentStore(str(tmp_path / "instances.sqlite3"))
    set_document_store(documents)
    instances = TaskInstanceStore()
    parent = instances.create_instance("parent", 1, total_steps=1)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    late_failures: list[BaseException] = []
    update = documents.update_one_and_get
    replace = documents.replace

    def delayed_update(collection, match, changes, **kwargs):
        if changes.set.get("context.metadata.owner") == "old":
            entered.set()
            wait_for_release(release)
        return update(collection, match, changes, **kwargs)

    def delayed_replace(collection, match, document, **kwargs):
        if document.get("step_id") == "__seed__" and document.get("generation") == 0:
            entered.set()
            wait_for_release(release)
        return replace(collection, match, document, **kwargs)

    monkeypatch.setattr(documents, "update_one_and_get", delayed_update if delayed_stage == "context" else update)
    monkeypatch.setattr(documents, "replace", delayed_replace if delayed_stage == "journal" else replace)

    def seed_old_attempt() -> None:
        try:
            instances.seed_context(parent.instance_id, ContextUpdateOps(sets={"context.metadata.owner": "old"}), attempt_epoch=0)
        except BaseException as error:
            late_failures.append(error)
            raise
        finally:
            finished.set()

    pending = asyncio.create_task(call_store(seed_old_attempt,
                                             deadline=asyncio.get_running_loop().time() + CALL_DEADLINE_SECONDS))
    try:
        assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
        with pytest.raises(StoreDeadlineExceeded):
            await pending
        fresh = instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
        await call_store(partial(instances.seed_context, parent.instance_id,
                                 ContextUpdateOps(sets={"context.metadata.owner": "fresh"}), attempt_epoch=fresh.attempt_epoch),
                         deadline=asyncio.get_running_loop().time() + TEST_DEADLINE_SECONDS)
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(pending, return_exceptions=True)

    assert len(late_failures) == 1 and isinstance(late_failures[0], SupersededAttemptError)
    assert instances.get(parent.instance_id).context["metadata"]["owner"] == "fresh"
    assert instances.replay_completed_context_sync(parent.instance_id, set())["metadata"] == {"owner": "fresh"}


@pytest.mark.asyncio
async def test_delayed_registrar_cannot_claim_an_epoch_after_a_newer_registration_commits(tmp_path, monkeypatch) -> None:
    documents = LocalSqliteDocumentStore(str(tmp_path / "instances.sqlite3"))
    set_document_store(documents)
    instances = TaskInstanceStore()
    parent = instances.create_instance("parent", 1, total_steps=1)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    old_thread: int | None = None
    update = documents.update_one_and_get

    def delayed_claim(collection, match, changes, **kwargs):
        if threading.get_ident() == old_thread and changes.inc.get("attempt_epoch") == 1:
            entered.set()
            wait_for_release(release)
        return update(collection, match, changes, **kwargs)

    monkeypatch.setattr(documents, "update_one_and_get", delayed_claim)

    def register_old_attempt():
        nonlocal old_thread
        old_thread = threading.get_ident()
        try:
            return instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
        finally:
            finished.set()

    pending = asyncio.create_task(call_store(register_old_attempt,
                                             deadline=asyncio.get_running_loop().time() + TEST_DEADLINE_SECONDS))
    try:
        assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
        fresh = instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
        assert fresh.attempt_epoch == 1
        release.set()
        with pytest.raises(SupersededAttemptError):
            await pending
        assert instances.get(parent.instance_id).attempt_epoch == fresh.attempt_epoch
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["deadline", "cancelled"])
async def test_abandoned_registrar_does_not_begin_a_claim_after_its_delayed_read_finishes(
    tmp_path, monkeypatch, abandonment: str,
) -> None:
    documents = LocalSqliteDocumentStore(str(tmp_path / "instances.sqlite3"))
    set_document_store(documents)
    instances = TaskInstanceStore()
    parent = instances.create_instance("parent", 1, total_steps=1)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    old_thread: int | None = None
    late_errors: list[BaseException] = []
    find = documents.find_one

    def delayed_read(collection, match, **kwargs):
        result = find(collection, match, **kwargs)
        if threading.get_ident() == old_thread:
            entered.set()
            wait_for_release(release)
        return result

    monkeypatch.setattr(documents, "find_one", delayed_read)

    def register_old_attempt():
        nonlocal old_thread
        old_thread = threading.get_ident()
        try:
            return instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
        except BaseException as error:
            late_errors.append(error)
            raise
        finally:
            finished.set()

    duration = CALL_DEADLINE_SECONDS if abandonment == "deadline" else TEST_DEADLINE_SECONDS
    pending = asyncio.create_task(call_store(register_old_attempt, deadline=asyncio.get_running_loop().time() + duration))
    try:
        assert await asyncio.to_thread(entered.wait, BACKEND_RELEASE_SECONDS)
        if abandonment == "cancelled":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(StoreDeadlineExceeded):
                await pending
        fresh = instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
        assert fresh.attempt_epoch == 1
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, BACKEND_RELEASE_SECONDS)
        await asyncio.gather(pending, return_exceptions=True)
    assert len(late_errors) == 1 and isinstance(late_errors[0], StoreCallAbandoned)
    assert instances.get(parent.instance_id).attempt_epoch == fresh.attempt_epoch


def test_registering_a_legacy_instance_without_attempt_epoch_claims_epoch_one(tmp_path) -> None:
    documents = LocalSqliteDocumentStore(str(tmp_path / "instances.sqlite3"))
    set_document_store(documents)
    instances = TaskInstanceStore()
    parent = instances.create_instance("parent", 1, total_steps=1)
    documents.update(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=parent.instance_id), UpdateSpec(unset={"attempt_epoch"}))
    fresh = instances.upsert_instance(parent.instance_id, "parent", 1, total_steps=1)
    assert fresh.attempt_epoch == 1
    assert instances.get(parent.instance_id).attempt_epoch == 1


@pytest.mark.parametrize("error", [StoreDeadlineExceeded("deadline"), StoreCallAbandoned("cancelled"),
                                  SupersededAttemptError("parent", "__seed__", "new owner")])
def test_registration_never_swallows_lifecycle_or_ownership_failure_without_a_caller_id(monkeypatch, error) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(TaskInstanceStore, "create_instance", fail)
    with pytest.raises(type(error)) as failure:
        register_task_instance("parent", 1, 1)
    assert failure.value is error
