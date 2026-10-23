"""Walk declared embedded task definitions without resolving them as stored task references."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import NamedTuple

from agent_env.task_step.task_step import TaskStep


class StepDefinition(NamedTuple):
    path: str
    value: dict
    step_class: type[TaskStep] | None
    label: str
    task_path: str


def embedded_tasks(step: dict, step_class: type[TaskStep]) -> Iterator[tuple[str, dict]]:
    for field in step_class.embedded_task_fields:
        definition = step.get(field)
        if not isinstance(definition, dict) or not isinstance(definition.get("steps"), list):
            raise ValueError(f"{field} must be an embedded task definition with a steps list")
        if any(not isinstance(child, dict) for child in definition["steps"]):
            raise ValueError(f"{field}.steps must contain step definitions")
        yield field, definition


def walk_step_definitions(
    steps: list[dict], registry: Mapping[str, type[TaskStep]], *, prefix: str = "steps", label_prefix: str = "",
) -> Iterator[StepDefinition]:
    for index, step in enumerate(steps):
        path = f"{prefix}[{index}]"
        step_type = step.get("type")
        step_class = registry.get(step_type) if isinstance(step_type, str) else None
        label = f"{label_prefix}step {step.get('id')!r}"
        yield StepDefinition(path, step, step_class, label, prefix)
        if step_class is not None:
            for field, definition in embedded_tasks(step, step_class):
                yield from walk_step_definitions(
                    definition["steps"], registry, prefix=f"{path}.{field}.steps", label_prefix=f"{label}: {field}: ",
                )
