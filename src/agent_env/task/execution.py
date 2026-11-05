"""The persisted step attempt owning a nested invocation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from collections.abc import Iterator


class TaskLifecycleError(RuntimeError):
    """A task cannot safely retry or tolerate an incomplete resource lifecycle."""


class TaskPersistenceFailure(TaskLifecycleError):
    """A required task persistence operation failed before its results could be published."""


class ContextWriteConflict(TaskLifecycleError):
    """Concurrent steps proposed incompatible writes to the same context path."""


@dataclass(frozen=True)
class StepExecution:
    instance_id: str | None
    step_id: str
    attempt_epoch: int


_CURRENT_STEP: ContextVar[StepExecution | None] = ContextVar("agent_env_current_step", default=None)


@contextmanager
def executing_step(execution: StepExecution) -> Iterator[None]:
    token = _CURRENT_STEP.set(execution)
    try:
        yield
    finally:
        _CURRENT_STEP.reset(token)


def current_step_execution() -> StepExecution:
    execution = _CURRENT_STEP.get()
    if execution is None or execution.instance_id is None:
        raise RuntimeError("Nested tasks require a persisted parent Task.run()")
    return execution
