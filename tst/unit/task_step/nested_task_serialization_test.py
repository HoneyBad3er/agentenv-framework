"""Nested task definitions retain their graph, bindings, budgets and exact JSON predicates."""

import copy

import pytest

from agent_env.task.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import RetryConfig
from agent_env.task_step.task_steps.repeat_until import RepeatUntilTaskStep, json_values_equal
from agent_env.task_step.task_steps.review import ReviewTaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep


def child_task() -> Task:
    return Task(
        id="child", version=4,
        steps=[ReviewTaskStep(id="review", version=2, label="accept", timeout_seconds=3, poll_interval_seconds=1)],
    )


def step_kwargs() -> dict:
    return {
        "id": "nested", "version": 2, "task": child_task(),
        "child_timeout_seconds": 10, "cleanup_timeout_seconds": 5,
        "input_bindings": {"candidate": "draft"}, "output_bindings": {"accepted": "result"},
    }


@pytest.mark.parametrize("step_class", [SubDAGTaskStep, RepeatUntilTaskStep])
def test_registered_nested_step_roundtrip_preserves_child_graph_bindings_budgets_and_retry(step_class):
    kwargs = step_kwargs()
    if step_class is RepeatUntilTaskStep:
        kwargs.update(max_iterations=3, until_key="done", until_value={"ok": [True, None]},
                      feedback_bindings={"candidate": "revision"})
    step = step_class(**kwargs, depends_on=["before"], fail_task_on_error=False,
                      retry_config=RetryConfig(retry_from_step_id="nested", max_retries=2))

    encoded = step.to_dict()
    restored = get_task_step_registry()[encoded["type"]].from_dict(encoded)

    assert type(restored) is step_class
    assert restored.to_dict() == encoded
    assert restored.task["version"] == 4
    assert restored.task["steps"][0]["version"] == 2
    assert restored.retry_config == RetryConfig(retry_from_step_id="nested", max_retries=2)
    assert restored.child_timeout_seconds == 10
    assert restored.cleanup_timeout_seconds == 5


def test_outer_task_roundtrip_keeps_arbitrarily_nested_step_definitions():
    leaf = SubDAGTaskStep(**step_kwargs())
    repeated = RepeatUntilTaskStep(
        **{**step_kwargs(), "task": Task(id="middle", version=None, steps=[leaf])},
        max_iterations=2, until_key="done", until_value=True,
    )
    outer = Task(id="outer", version=None, steps=[repeated])

    assert Task.from_dict(outer.to_dict()).to_dict() == outer.to_dict()


def test_embedded_task_snapshot_is_not_changed_by_mutating_original_or_serialized_graph():
    task = child_task()
    step = SubDAGTaskStep(**{**step_kwargs(), "task": task})
    snapshot = copy.deepcopy(step.task)

    task.steps[0].label = "changed"
    encoded = step.to_dict()
    encoded["task"]["steps"][0]["label"] = "also changed"

    assert step.task == snapshot


@pytest.mark.parametrize("field", ["child_timeout_seconds", "cleanup_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "3", None])
def test_nested_step_rejects_unbounded_or_invalid_timeout(field, value):
    with pytest.raises(ValueError, match=field):
        SubDAGTaskStep(**{**step_kwargs(), field: value})


@pytest.mark.parametrize("field", ["child_timeout_seconds", "cleanup_timeout_seconds"])
def test_nested_step_deserialization_requires_each_timeout_budget(field):
    encoded = SubDAGTaskStep(**step_kwargs()).to_dict()
    del encoded[field]

    with pytest.raises(KeyError, match=field):
        SubDAGTaskStep.from_dict(encoded)


@pytest.mark.parametrize("task", [
    None, "child", {}, {"id": "child", "steps": []}, {"id": "", "steps": [{}]},
    {"id": "child", "type": "unknown", "steps": [{}]},
])
def test_nested_step_rejects_missing_empty_or_wrong_type_child_definition(task):
    with pytest.raises(ValueError):
        SubDAGTaskStep(**{**step_kwargs(), "task": task})


def test_nested_step_rejects_invalid_child_dependency_graph_before_execution():
    definition = child_task().to_dict()
    definition["steps"][0]["depends_on"] = ["missing"]

    with pytest.raises(ValueError, match="names no step"):
        SubDAGTaskStep(**{**step_kwargs(), "task": definition})


@pytest.mark.parametrize("field", ["input_bindings", "output_bindings"])
@pytest.mark.parametrize("bindings", [
    [], {"": "value"}, {"value": ""}, {"a.b": "value"}, {"value": "a.b"},
    {"$set": "value"}, {"value": "$set"}, {"user_overrides": "value"},
    {"value": "failed_steps"}, {"litellm_api_key": "value"}, {"value": "_heartbeat_fn"},
    {"child_runs": "value"}, {"seed": "value"}, {"value": 3},
    {"parent_task": "value"}, {"value": "parent_task"},
])
def test_nested_bindings_reject_paths_framework_keys_secret_keys_and_invalid_maps(field, bindings):
    with pytest.raises(ValueError, match=field):
        SubDAGTaskStep(**{**step_kwargs(), field: bindings})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_repeat_until_rejects_missing_or_unbounded_iteration_budget(value):
    with pytest.raises(ValueError, match="max_iterations"):
        RepeatUntilTaskStep(**step_kwargs(), max_iterations=value, until_key="done", until_value=True)


@pytest.mark.parametrize("field, value", [
    ("until_key", "nested.done"), ("until_key", "failed_steps"),
    ("until_value", float("nan")), ("until_value", {"n": float("inf")}),
    ("until_value", (True,)), ("until_value", {3: True}),
    ("until_value", [{"litellm_api_key": "secret value"}]),
    ("feedback_bindings", {"candidate": "user_overrides"}),
])
def test_repeat_until_rejects_unsafe_predicate_or_feedback(field, value):
    kwargs = {**step_kwargs(), "max_iterations": 2, "until_key": "done", "until_value": True}
    with pytest.raises(ValueError, match=field):
        RepeatUntilTaskStep(**{**kwargs, field: value})


@pytest.mark.parametrize("left,right,expected", [
    (True, 1, False), (False, 0, False), ({"a": [True]}, {"a": [1]}, False),
    ({"a": [False]}, {"a": [False]}, True), ({"a": 1, "b": None}, {"b": None, "a": 1}, True),
    (None, None, True), ([1, 2], [2, 1], False), (1, 1.0, False),
])
def test_repeat_condition_compares_json_types_recursively(left, right, expected):
    assert json_values_equal(left, right) is expected
