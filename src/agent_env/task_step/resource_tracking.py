"""Checkpoint resource ownership without publishing a step's domain outputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_env.task_step.context import TaskStepContext


TORN_DOWN_RESOURCE_KEY = "torn_down_sandbox_ids"

ResourceCheckpoint = Callable[["TaskStepContext"], Awaitable[None]]
_CHECKPOINT: ContextVar[ResourceCheckpoint | None] = ContextVar("agent_env_resource_checkpoint", default=None)


@contextmanager
def tracking_resources(checkpoint: ResourceCheckpoint) -> Iterator[None]:
    token = _CHECKPOINT.set(checkpoint)
    try:
        yield
    finally:
        _CHECKPOINT.reset(token)


async def checkpoint_resources(context: TaskStepContext) -> None:
    checkpoint = _CHECKPOINT.get()
    if checkpoint is not None:
        await checkpoint(context)
