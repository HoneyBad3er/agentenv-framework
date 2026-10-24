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

from agent_env.task.execution import TaskLifecycleError

DEFAULT_STORE_TIMEOUT_SECONDS = 30.0
MAX_CONCURRENT_STORE_CALLS = 8
_STORE_SLOT_POLL_SECONDS = 0.01
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STORE_CALLS, thread_name_prefix="agentenv-persistence")
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_STORE_CALLS)
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("agentenv_store_deadline", default=None)
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
    return inherited if inherited is not None else asyncio.get_running_loop().time() + DEFAULT_STORE_TIMEOUT_SECONDS


def _observe_future(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def _invoke(operation: Callable[[], _T]) -> _T:
    check_store_call()
    return operation()


async def call_store(operation: Callable[[], _T], *, deadline: float) -> _T:
    """Preserve contextvars off-loop; a slot stays occupied until the backend call actually stops, even after timeout."""
    loop = asyncio.get_running_loop()
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
