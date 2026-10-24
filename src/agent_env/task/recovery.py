"""Recover recorded child resources after their owner stopped executing."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING

from agent_env.store.base import NotFoundError
from agent_env.store.routing import run_scope
from agent_env.task.child_store import ChildCleanupState, get_child_invocation_store
from agent_env.task.execution import TaskLifecycleError
from agent_env.task.persistence import call_store
from agent_env.task.resources import load_resource_context
from agent_env.task.store import get_task_instance_store
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY, _env_sandbox_ids

if TYPE_CHECKING:
    from agent_env.task.teardown import TeardownReport


def merge_resource_contexts(*contexts: TaskStepContext) -> TaskStepContext:
    merged = TaskStepContext(metadata=copy.deepcopy(contexts[0].metadata) if contexts else {})
    merged.metadata.pop("retry_resets", None)
    merged.metadata.pop("agent_changelog", None)
    merged.metadata.pop(TORN_DOWN_KEY, None)
    resources = {field: {} for field in ("deployed_envs", "deployed_agents", "deployed_sandboxes")}
    for context in contexts:
        for field, records in resources.items():
            for resource in getattr(context, field):
                key = json.dumps({
                    "sandbox_id": resource.sandbox_id, "sandbox_ids": getattr(resource, "sandbox_ids", None),
                    "name": getattr(resource, "agent_name", None) if not resource.sandbox_id else None,
                }, sort_keys=True)
                records[key] = copy.deepcopy(resource)
        for field in ("retry_resets", "agent_changelog"):
            entries = {
                json.dumps(entry, sort_keys=True): entry
                for entry in [*(merged.metadata.get(field) or []), *(context.metadata.get(field) or [])]
            }
            if entries:
                merged.metadata[field] = copy.deepcopy(list(entries.values()))
        if terminated := context.metadata.get(TORN_DOWN_KEY):
            merged.metadata[TORN_DOWN_KEY] = sorted(set(merged.metadata.get(TORN_DOWN_KEY, ())) | set(terminated))
        if context.instance_id:
            merged.instance_id = context.instance_id
    for field, records in resources.items():
        setattr(merged, field, list(records.values()))
    return merged


def pending_resource_context(context: TaskStepContext) -> TaskStepContext:
    pending = copy.deepcopy(context)
    terminated = set(context.metadata.get(TORN_DOWN_KEY) or ())
    pending.deployed_sandboxes = [resource for resource in pending.deployed_sandboxes if resource.sandbox_id not in terminated]
    pending.deployed_agents = [resource for resource in pending.deployed_agents if resource.sandbox_id not in terminated]
    pending.deployed_envs = [
        resource for resource in pending.deployed_envs
        if any(sandbox_id not in terminated for sandbox_id, _ in _env_sandbox_ids(resource))
    ]
    return pending


async def recover_child_tree(
    *, parent_instance_id: str, timeout_seconds: float,
    cleanup: Callable[[TaskStepContext, float], Awaitable[TeardownReport]],
    deadline: float | None = None,
) -> tuple[TeardownReport, ...]:
    deadline = deadline if deadline is not None else asyncio.get_running_loop().time() + timeout_seconds
    store = get_child_invocation_store()
    with run_scope(parent_instance_id):
        children = await call_store(partial(store.list_for_parent, parent_instance_id), deadline=deadline)
        if not children:
            return ()
        parent = await call_store(partial(get_task_instance_store().get, parent_instance_id), deadline=deadline)
        reports = []
        for invocation in children:
            reports.extend(await recover_child_tree(
                parent_instance_id=invocation.child_instance_id, timeout_seconds=timeout_seconds,
                cleanup=cleanup, deadline=deadline,
            ))
            if invocation.cleanup_state is ChildCleanupState.COMPLETED:
                continue
            child_id = invocation.child_instance_id
            with run_scope(invocation.task_definition["id"]):
                try:
                    instance = await call_store(partial(get_task_instance_store().get, child_id), deadline=deadline)
                except NotFoundError:
                    context = TaskStepContext()
                else:
                    context = TaskStepContext.from_dict(copy.deepcopy(instance.context or {}))
                inventory = await load_resource_context(child_id, deadline=deadline)
            captured = TaskStepContext.from_dict(copy.deepcopy(invocation.cleanup_context or {}))
            context = merge_resource_contexts(context, captured, inventory)
            context.instance_id = child_id
            invocation = await call_store(partial(
                store.recover_cleanup_state, invocation, parent_attempt_epoch=parent.attempt_epoch,
                state=ChildCleanupState.RUNNING,
            ), deadline=deadline)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TaskLifecycleError("Child resource recovery deadline expired")
            try:
                async with asyncio.timeout(remaining):
                    report = await cleanup(context, remaining)
                state = ChildCleanupState.FAILED if report.still_up else ChildCleanupState.COMPLETED
                error = "child resources remain after recovery" if report.still_up else None
                await call_store(partial(
                    store.recover_cleanup_state, invocation, parent_attempt_epoch=parent.attempt_epoch,
                    state=state, error=error, context=context.to_safe_dict(),
                ), deadline=asyncio.get_running_loop().time() + timeout_seconds)
            except BaseException as error:
                try:
                    await call_store(partial(
                        store.recover_cleanup_state, invocation, parent_attempt_epoch=parent.attempt_epoch,
                        state=ChildCleanupState.FAILED, error=type(error).__name__, context=context.to_safe_dict(),
                    ), deadline=asyncio.get_running_loop().time() + timeout_seconds)
                except Exception as recording_error:
                    error.add_note(f"Recording resource recovery failure also failed: {type(recording_error).__name__}")
                raise
            reports.append(report)
        return tuple(reports)
