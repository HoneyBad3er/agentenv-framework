"""Run a bounded sequence of isolated child tasks until a JSON predicate matches."""

from __future__ import annotations

from typing import Any, ClassVar

from agent_env.task.nested import run_child
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStepDependency
from agent_env.task_step.task_steps.subdag import (
    SubDAGTaskStep,
    copy_json_value,
    read_metadata,
    validate_bindings,
    validate_metadata_key,
)


class RepeatUntilExhaustedError(RuntimeError):
    """The child predicate did not match within the configured iteration budget."""


def json_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(json_values_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(json_values_equal(a, b) for a, b in zip(left, right))
    return left == right


class RepeatUntilTaskStep(SubDAGTaskStep):
    type: ClassVar[str] = "repeat_until"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: int | None,
        *,
        task: Task | dict,
        child_timeout_seconds: float,
        cleanup_timeout_seconds: float,
        max_iterations: int,
        until_key: str,
        until_value: Any,
        input_bindings: dict[str, str] | None = None,
        output_bindings: dict[str, str] | None = None,
        feedback_bindings: dict[str, str] | None = None,
        depends_on: list[TaskStepDependency] | None = None,
        fail_task_on_error: bool = True,
        retry_config: RetryConfig | dict | None = None,
    ):
        super().__init__(
            id=id, version=version, task=task, child_timeout_seconds=child_timeout_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            input_bindings=input_bindings, output_bindings=output_bindings, depends_on=depends_on,
            fail_task_on_error=fail_task_on_error, retry_config=retry_config,
        )
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = max_iterations
        self.until_key = validate_metadata_key(until_key, field="until_key")
        self.until_value = copy_json_value(until_value, field="until_value")
        self.feedback_bindings = validate_bindings(feedback_bindings, field="feedback_bindings")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "max_iterations": self.max_iterations,
            "until_key": self.until_key,
            "until_value": copy_json_value(self.until_value, field="until_value"),
            "feedback_bindings": dict(self.feedback_bindings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepeatUntilTaskStep:
        return cls(
            **cls._base_from_dict(data), task=data["task"],
            child_timeout_seconds=data["child_timeout_seconds"],
            cleanup_timeout_seconds=data["cleanup_timeout_seconds"],
            max_iterations=data["max_iterations"], until_key=data["until_key"], until_value=data["until_value"],
            input_bindings=data.get("input_bindings"), output_bindings=data.get("output_bindings"),
            feedback_bindings=data.get("feedback_bindings"), retry_config=data.get("retry_config"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        base_inputs = read_metadata(context.metadata, self.input_bindings, field="input_bindings")
        feedback = {}
        for iteration in range(self.max_iterations):
            child_context = await run_child(
                step=self, context=context, task=self.task,
                inputs={**base_inputs, **feedback}, iteration=iteration,
                child_timeout_seconds=self.child_timeout_seconds,
                cleanup_timeout_seconds=self.cleanup_timeout_seconds,
                step_params=self.step_param_overrides(context).get("child_step_params", {}).get("task"),
            )
            if self.until_key not in child_context.metadata:
                raise ValueError(f"until_key metadata {self.until_key!r} is missing from the child task")
            observed = copy_json_value(child_context.metadata[self.until_key], field="until_key")
            if json_values_equal(observed, self.until_value):
                outputs = read_metadata(child_context.metadata, self.output_bindings, field="output_bindings")
                context.metadata.update(outputs)
                return context
            if iteration + 1 < self.max_iterations:
                feedback = read_metadata(child_context.metadata, self.feedback_bindings, field="feedback_bindings")
        raise RepeatUntilExhaustedError(f"{self.id}: condition did not match after {self.max_iterations} iterations")
