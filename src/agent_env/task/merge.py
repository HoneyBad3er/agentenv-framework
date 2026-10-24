"""Publish isolated step changes without importing a concurrent sibling's writes."""

from __future__ import annotations

import copy
import dataclasses

from agent_env.task.execution import ContextWriteConflict
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.context_ops import _has_unmappable_keys
from agent_env.task_step.resource_tracking import TORN_DOWN_RESOURCE_KEY


_ABSENT = object()
_RESOURCE_FIELDS = ("deployed_envs", "deployed_agents", "deployed_sandboxes")


def _resource_key(name: str, resource):
    if name == "deployed_envs":
        return resource.instance_id or resource
    if name == "deployed_agents":
        return resource.agent_name, resource.sandbox_id, resource.sandbox_type
    return resource.sandbox_name, resource.sandbox_id, resource.sandbox_type


def _merge_value(before, after, current, *, path: str):
    if after is not _ABSENT and before is not _ABSENT and after == before:
        return current
    if after is _ABSENT:
        if current is _ABSENT or current == before:
            return _ABSENT
        raise ContextWriteConflict(f"Concurrent writes conflict at {path}")
    atomic_container = before is _ABSENT and (after == {} or after == [] or current == {} or current == [])
    atomic_mapping = any(isinstance(value, dict) and _has_unmappable_keys(value) for value in (before, after, current))
    if isinstance(after, dict) and (before is _ABSENT or isinstance(before, dict)) and not (atomic_container or atomic_mapping):
        if current is _ABSENT:
            if before is not _ABSENT:
                raise ContextWriteConflict(f"Concurrent writes conflict at {path}")
            current = {}
        if not isinstance(current, dict):
            raise ContextWriteConflict(f"Concurrent writes conflict at {path}")
        baseline = {} if before is _ABSENT else before
        merged = copy.deepcopy(current)
        for key in dict.fromkeys((*baseline, *after)):
            value = _merge_value(baseline.get(key, _ABSENT), after.get(key, _ABSENT), current.get(key, _ABSENT),
                                 path=f"{path}.{key}")
            if value is _ABSENT:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged
    if isinstance(after, list) and (before is _ABSENT or isinstance(before, list)) and not atomic_container:
        baseline = [] if before is _ABSENT else before
        if after[:len(baseline)] == baseline:
            if current is _ABSENT:
                current = []
            if not isinstance(current, list) or current[:len(baseline)] != baseline:
                raise ContextWriteConflict(f"Concurrent writes conflict at {path}")
            merged = copy.deepcopy(current)
            for item in after[len(baseline):]:
                if path == "context.deployed_envs":
                    present = any(_resource_key("deployed_envs", item) == _resource_key("deployed_envs", existing)
                                  for existing in merged)
                else:
                    present = item in merged
                if not present:
                    merged.append(copy.deepcopy(item))
            return merged
    if current is _ABSENT and before is _ABSENT or current is not _ABSENT and (current == before or current == after):
        return copy.deepcopy(after)
    raise ContextWriteConflict(f"Concurrent writes conflict at {path}")


def merge_step_context(*, before: TaskStepContext, after: TaskStepContext, target: TaskStepContext) -> None:
    values = {
        field.name: _merge_value(getattr(before, field.name), getattr(after, field.name), getattr(target, field.name),
                                path=f"context.{field.name}")
        for field in dataclasses.fields(TaskStepContext)
    }
    for name, value in values.items():
        setattr(target, name, value)


def retain_step_resources(*, source: TaskStepContext, target: TaskStepContext) -> None:
    for name in _RESOURCE_FIELDS:
        retained = getattr(target, name)
        for resource in getattr(source, name):
            matching = next((index for index, existing in enumerate(retained)
                             if _resource_key(name, existing) == _resource_key(name, resource)), None)
            if matching is None:
                retained.append(copy.deepcopy(resource))
            else:
                retained[matching] = copy.deepcopy(resource)
    if terminated := source.metadata.get(TORN_DOWN_RESOURCE_KEY):
        recorded = target.metadata.setdefault(TORN_DOWN_RESOURCE_KEY, [])
        recorded.extend(identifier for identifier in terminated if identifier not in recorded)
    if changelogs := source.metadata.get("agent_changelog"):
        recorded = target.metadata.setdefault("agent_changelog", [])
        recorded.extend(copy.deepcopy(entry) for entry in changelogs if entry not in recorded)
