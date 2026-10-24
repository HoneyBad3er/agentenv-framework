"""Run an embedded task with explicit metadata inputs and outputs."""

from __future__ import annotations

import copy
import json
import math
from typing import Any, ClassVar

from agent_env.task.nested import run_child
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext, _REDACTED_KEYS, _strip_redacted_keys
from agent_env.task_step.resource_tracking import TORN_DOWN_RESOURCE_KEY
from agent_env.task_step.task_step import RetryConfig, TaskStep, TaskStepDependency


_FRAMEWORK_METADATA_KEYS = frozenset({
    "user_overrides", "failed_steps", "child_runs", "runtime", "seed", "workflow_id",
    "run_group_id", "pass_at_k_execution_id", "instance_id", "start_step", "retry_resets",
    "models", "step_overrides", "step_params", "parent_task", *_REDACTED_KEYS,
    "keep", "runner", "episode", "agent_changelog", "agent_changelog_rewinds", "agent_loaded_snapshots",
    TORN_DOWN_RESOURCE_KEY,
})


def validate_metadata_key(key: str, *, field: str) -> str:
    if not isinstance(key, str) or not key or "." in key or key.startswith("$"):
        raise ValueError(f"{field} must name a nonempty metadata key without dots or a leading '$'")
    if key in _FRAMEWORK_METADATA_KEYS:
        raise ValueError(f"{field} cannot name framework metadata key {key!r}")
    return key


def validate_bindings(bindings: dict[str, str] | None, *, field: str) -> dict[str, str]:
    if bindings is None:
        return {}
    if not isinstance(bindings, dict):
        raise ValueError(f"{field} must map metadata keys to metadata keys")
    return {
        validate_metadata_key(target, field=field): validate_metadata_key(source, field=field)
        for target, source in bindings.items()
    }


def copy_json_value(value: Any, *, field: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError(f"{field} must contain only finite JSON values") from error

    def validate(candidate: Any) -> None:
        if candidate is None or isinstance(candidate, (str, bool, int)):
            return
        if isinstance(candidate, float) and math.isfinite(candidate):
            return
        if isinstance(candidate, list):
            for item in candidate:
                validate(item)
            return
        if isinstance(candidate, dict) and all(isinstance(key, str) for key in candidate):
            for item in candidate.values():
                validate(item)
            return
        raise ValueError(f"{field} must contain only finite JSON values")

    validate(value)
    if _strip_redacted_keys(value) != value:
        raise ValueError(f"{field} cannot contain secret metadata")
    return copy.deepcopy(value)


def read_metadata(metadata: dict, bindings: dict[str, str], *, field: str) -> dict:
    values = {}
    for target, source in bindings.items():
        if source not in metadata:
            raise ValueError(f"{field} source metadata key {source!r} is missing")
        values[target] = copy_json_value(metadata[source], field=f"{field}.{source}")
    return values


class SubDAGTaskStep(TaskStep):
    type: ClassVar[str] = "subdag"
    entity_refs = ()
    embedded_task_fields = ("task",)

    def __init__(
        self,
        id: str,
        version: int | None,
        *,
        task: Task | dict,
        child_timeout_seconds: float,
        cleanup_timeout_seconds: float,
        input_bindings: dict[str, str] | None = None,
        output_bindings: dict[str, str] | None = None,
        depends_on: list[TaskStepDependency] | None = None,
        fail_task_on_error: bool = True,
        retry_config: RetryConfig | dict | None = None,
    ):
        super().__init__(
            id=id, version=version, depends_on=depends_on,
            fail_task_on_error=fail_task_on_error, retry_config=retry_config,
        )
        for name, timeout in (
            ("child_timeout_seconds", child_timeout_seconds),
            ("cleanup_timeout_seconds", cleanup_timeout_seconds),
        ):
            if (
                isinstance(timeout, bool) or not isinstance(timeout, (float, int))
                or not math.isfinite(timeout) or timeout <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        raw_task = task.to_dict() if isinstance(task, Task) else task
        if not isinstance(raw_task, dict):
            raise ValueError("task must be an embedded Task definition")
        raw_task = copy_json_value(raw_task, field="task")
        if raw_task.get("type", Task.type) != Task.type:
            raise ValueError("task must have type 'task'")
        if not isinstance(raw_task.get("id"), str) or not raw_task["id"]:
            raise ValueError("task must have a nonempty id")
        if not isinstance(raw_task.get("steps"), list) or not raw_task["steps"]:
            raise ValueError("task must contain at least one step")
        self.task = copy_json_value(Task.from_dict(raw_task).to_dict(), field="task")
        self.child_timeout_seconds = float(child_timeout_seconds)
        self.cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        self.input_bindings = validate_bindings(input_bindings, field="input_bindings")
        self.output_bindings = validate_bindings(output_bindings, field="output_bindings")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "task": copy.deepcopy(self.task),
            "child_timeout_seconds": self.child_timeout_seconds,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "input_bindings": dict(self.input_bindings),
            "output_bindings": dict(self.output_bindings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SubDAGTaskStep:
        return cls(
            **cls._base_from_dict(data), task=data["task"],
            child_timeout_seconds=data["child_timeout_seconds"],
            cleanup_timeout_seconds=data["cleanup_timeout_seconds"],
            input_bindings=data.get("input_bindings"), output_bindings=data.get("output_bindings"),
            retry_config=data.get("retry_config"),
        )

    def config_problems(self) -> list[str]:
        return [
            f"{self.id}: {problem}"
            for child_step in Task.from_dict(self.task).steps
            for problem in child_step.config_problems()
        ]

    def preflight(self) -> list[str]:
        return [
            f"{self.id}: {problem}"
            for child_step in Task.from_dict(self.task).steps
            for problem in child_step.preflight()
        ]

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        child_context = await run_child(
            step=self, context=context, task=self.task,
            inputs=read_metadata(context.metadata, self.input_bindings, field="input_bindings"),
            iteration=0, child_timeout_seconds=self.child_timeout_seconds,
            cleanup_timeout_seconds=self.cleanup_timeout_seconds,
        )
        outputs = read_metadata(child_context.metadata, self.output_bindings, field="output_bindings")
        context.metadata.update(outputs)
        return context
