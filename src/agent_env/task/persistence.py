"""Bound persistence callers while keeping timed-out synchronous work inside a fixed process-wide capacity."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from agent_env.task.execution import TaskLifecycleError, TaskPersistenceFailure

DEFAULT_STORE_TIMEOUT_SECONDS = 30.0
MAX_CONCURRENT_STORE_CALLS = 8
_STORE_SLOT_POLL_SECONDS = 0.01
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STORE_CALLS, thread_name_prefix="agentenv-persistence")
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_STORE_CALLS)
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("agentenv_store_deadline", default=None)
_required: contextvars.ContextVar[bool] = contextvars.ContextVar("agentenv_task_persistence_required", default=False)
_timeout_seconds: contextvars.ContextVar[float] = contextvars.ContextVar(
    "agentenv_store_timeout_seconds", default=DEFAULT_STORE_TIMEOUT_SECONDS,
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class _StoreCall:
    deadline: float
    abandoned: threading.Event


_call: contextvars.ContextVar[_StoreCall | None] = contextvars.ContextVar("agentenv_store_call", default=None)


class StoreDeadlineExceeded(TaskLifecycleError, TimeoutError):
    """Persistence did not finish within its caller's deadline; a running backend call can still be finishing."""


class StoreCallAbandoned(TaskLifecycleError):
    """The awaiting task has left this synchronous operation; it must not begin another mutation."""


@contextmanager
def require_task_persistence(*, required: bool, timeout_seconds: float | None = None) -> Iterator[None]:
    """A nested run inherits its parent's persistence requirement and cannot weaken it."""
    token = _required.set(required or _required.get())
    timeout_token = _timeout_seconds.set(timeout_seconds if timeout_seconds is not None else _timeout_seconds.get())
    try:
        yield
    finally:
        _timeout_seconds.reset(timeout_token)
        _required.reset(token)


def is_task_persistence_required() -> bool:
    return _required.get()


def raise_persistence_failure(error: Exception, *, operation: str) -> None:
    if is_task_persistence_required():
        raise TaskPersistenceFailure(f"Required task persistence failed: {operation}") from error


def check_store_call() -> None:
    active = _call.get()
    if active is None:
        return
    if active.abandoned.is_set():
        raise StoreCallAbandoned("the persistence caller has abandoned this operation")
    if time.monotonic() >= active.deadline:
        raise StoreDeadlineExceeded("the persistence deadline expired before a mutation")


@contextmanager
def store_deadline(deadline: float) -> Iterator[None]:
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def current_store_deadline() -> float:
    """Use the execution's deadline, or a bounded call budget for Task.run(), which has no overall deadline."""
    inherited = _deadline.get()
    if inherited is not None and _call.get() is not None:
        return inherited
    deadline = asyncio.get_running_loop().time() + _timeout_seconds.get()
    return min(inherited, deadline) if inherited is not None else deadline


def _observe_future(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def _invoke(operation: Callable[[], _T]) -> _T:
    check_store_call()
    return operation()


async def call_store(operation: Callable[[], _T], *, deadline: float) -> _T:
    """Preserve contextvars off-loop; a slot stays occupied until the backend call actually stops, even after timeout."""
    loop = asyncio.get_running_loop()
    deadline = min(deadline, loop.time() + _timeout_seconds.get())
    if deadline <= loop.time():
        raise StoreDeadlineExceeded("persistence deadline expired before the call started")
    active = _StoreCall(deadline=time.monotonic() + deadline - loop.time(), abandoned=threading.Event())
    budget = asyncio.timeout_at(deadline)
    try:
        async with budget:
            while not _slots.acquire(blocking=False):
                await asyncio.sleep(_STORE_SLOT_POLL_SECONDS)
            try:
                if deadline <= loop.time():
                    raise StoreDeadlineExceeded("persistence deadline expired while waiting for capacity")
                context = contextvars.copy_context()
                context.run(_call.set, active)
                context.run(_deadline.set, deadline)
                pending = _executor.submit(context.run, _invoke, operation)
            except BaseException:
                _slots.release()
                raise
            pending.add_done_callback(lambda future: _slots.release())
            result = asyncio.wrap_future(pending, loop=loop)
            result.add_done_callback(_observe_future)
            return await result
    except TimeoutError as error:
        if budget.expired():
            raise StoreDeadlineExceeded("persistence call exceeded its deadline") from error
        raise
    finally:
        active.abandoned.set()
