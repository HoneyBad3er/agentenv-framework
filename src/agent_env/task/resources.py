"""Persist in-flight resource ownership separately from committed task outputs."""

from __future__ import annotations

import copy
from functools import partial

from agent_env.config import get_config
from agent_env.store.document_store import Filter, UpdateSpec
from agent_env.store.routing import run_scope
from agent_env.task.merge import retain_step_resources
from agent_env.task.execution import TaskLifecycleError
from agent_env.task.persistence import call_store, raise_persistence_failure
from agent_env.task.store import TASK_INSTANCES_COLLECTION, SupersededAttemptError
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.resource_tracking import TORN_DOWN_RESOURCE_KEY


def _record_resource_context(context: TaskStepContext, *, attempt_epoch: int) -> None:
    if context.instance_id is None:
        raise RuntimeError("Resource checkpoints require a persisted task instance")
    store = get_config().get_document_store()
    safe = context.to_safe_dict()
    fields = ("deployed_envs", "deployed_agents", "deployed_sandboxes")
    additions = {f"resource_context.{name}": safe[name] for name in fields if safe[name]}
    if terminated := safe.get("metadata", {}).get(TORN_DOWN_RESOURCE_KEY):
        additions[f"resource_context.metadata.{TORN_DOWN_RESOURCE_KEY}"] = terminated
    if changelogs := safe.get("metadata", {}).get("agent_changelog"):
        additions["resource_context.metadata.agent_changelog"] = changelogs
    if not additions:
        return
    updated = store.update_one_and_get(
        TASK_INSTANCES_COLLECTION, Filter.of(instance_id=context.instance_id, attempt_epoch=attempt_epoch),
        UpdateSpec(add_to_set=additions, inc={"rev": 1}),
    )
    if updated is None:
        raise SupersededAttemptError(context.instance_id, None, "resource checkpoint no longer owns the task")


async def record_resource_context(context: TaskStepContext, *, attempt_epoch: int, deadline: float) -> None:
    if (context.deployed_envs or context.deployed_agents or context.deployed_sandboxes
            or context.metadata.get(TORN_DOWN_RESOURCE_KEY) or context.metadata.get("agent_changelog")):
        snapshot = copy.deepcopy(context)
        with run_scope(context.instance_id):
            try:
                await call_store(partial(_record_resource_context, snapshot, attempt_epoch=attempt_epoch), deadline=deadline)
            except (SupersededAttemptError, TaskLifecycleError):
                raise
            except Exception as error:
                raise_persistence_failure(error, operation="record resource ownership")
                raise


def _load_resource_context(instance_id: str) -> TaskStepContext:
    row = get_config().get_document_store().find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
    if row is None:
        return TaskStepContext()
    captured = TaskStepContext.from_dict(copy.deepcopy(row.get("resource_context") or {}))
    normalized = TaskStepContext(instance_id=instance_id)
    retain_step_resources(source=captured, target=normalized)
    return normalized


async def load_resource_context(instance_id: str, *, deadline: float) -> TaskStepContext:
    with run_scope(instance_id):
        try:
            return await call_store(partial(_load_resource_context, instance_id), deadline=deadline)
        except (SupersededAttemptError, TaskLifecycleError):
            raise
        except Exception as error:
            raise_persistence_failure(error, operation="load resource ownership")
            raise
