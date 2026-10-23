"""Execute isolated child graphs under durable parent step identities."""

from __future__ import annotations

import asyncio
import copy

from agent_env.runner.runner import RunStatus, run_outcome
from agent_env.store.base import NotFoundError
from agent_env.store.routing import run_scope
from agent_env.task.child_store import ChildCleanupState, ChildInvocation, get_child_invocation_store
from agent_env.task.execution import StepExecution, TaskLifecycleError, current_step_execution
from agent_env.task.store import SupersededAttemptError, TaskStepStatus, get_task_instance_store
from agent_env.task.task import Task
from agent_env.task.teardown import recorded_sandboxes, teardown_run
from agent_env.task_step.context import HEARTBEAT_KEY, TaskStepContext, regraft_redacted_keys
from agent_env.task_step.task_step import TaskStep


class ChildTaskFailed(RuntimeError):
    """A child graph did not produce a successful outcome."""

    def __init__(self, child_instance_id: str, reason: str) -> None:
        self.child_instance_id = child_instance_id
        super().__init__(f"Child task {child_instance_id}: {reason}")


class ChildTaskTimeout(ChildTaskFailed):
    """The configured child execution deadline expired."""


class ChildCleanupFailure(ChildTaskFailed, TaskLifecycleError):
    """Child resources could not be confirmed terminated."""


def _initial_context(
    *, parent: TaskStepContext, inputs: dict, execution: StepExecution, generation: int, iteration: int,
) -> TaskStepContext:
    metadata = copy.deepcopy(inputs)
    metadata["user_overrides"] = copy.deepcopy(parent.metadata.get("user_overrides") or {})
    metadata["user_overrides"].pop("step_params", None)
    metadata["parent_task"] = {
        "instance_id": execution.instance_id,
        "step_id": execution.step_id,
        "step_generation": generation,
        "iteration": iteration,
    }
    for key in ("workflow_id", "run_group_id"):
        if key in parent.metadata:
            metadata[key] = parent.metadata[key]
    if HEARTBEAT_KEY in parent.metadata:
        metadata[HEARTBEAT_KEY] = parent.metadata[HEARTBEAT_KEY]
    return TaskStepContext(
        metadata=metadata, agent_model=parent.agent_model, default_agent_model=parent.default_agent_model,
        agent_artifact_id=parent.agent_artifact_id, agent_harness=parent.agent_harness,
    )


def _completed_dependencies(task: Task, completed: set[str]) -> set[str]:
    state = task._build_scheduler_state(task.steps, 0)
    retained = set()
    for step in task.steps:
        if step.id in completed and state.dependencies[step.id] <= retained:
            retained.add(step.id)
    return retained


def _restore_context(
    task: Task, invocation: ChildInvocation, initial: TaskStepContext,
) -> tuple[TaskStepContext, set[str], bool, bool]:
    if invocation.outcome is not None:
        return TaskStepContext.from_dict(copy.deepcopy(invocation.outcome)), set(), True, False
    try:
        with run_scope(task.id):
            instance = get_task_instance_store().get(invocation.child_instance_id)
    except NotFoundError:
        return initial, set(), False, False
    if instance.task_id != task.id or instance.task_version != task.version:
        raise ChildTaskFailed(invocation.child_instance_id, "stored task identity changed")
    restored = copy.deepcopy(instance.context or {})
    regraft_redacted_keys(initial.to_dict(), restored)
    context = TaskStepContext.from_dict(restored)
    completed = {step.step_id for step in instance.completed_steps if step.status is TaskStepStatus.SUCCESS}
    status, _ = run_outcome(context)
    if instance.status == "completed" and status is RunStatus.COMPLETED:
        return context, completed, True, False
    if recorded_sandboxes(context):
        return initial, set(), False, True
    retained = _completed_dependencies(task, completed)
    with run_scope(task.id):
        restored = get_task_instance_store().replay_completed_context_sync(invocation.child_instance_id, retained)
    regraft_redacted_keys(initial.to_dict(), restored)
    context = TaskStepContext.from_dict(restored)
    for key in ("workflow_id", "run_group_id"):
        if key in initial.metadata:
            context.metadata[key] = initial.metadata[key]
    return context, retained, False, False


async def _clean_child(
    invocation: ChildInvocation, context: TaskStepContext, *, timeout_seconds: float,
) -> ChildInvocation:
    store = get_child_invocation_store()
    recording_error: Exception | None = None
    try:
        invocation = store.set_cleanup_state(invocation, state=ChildCleanupState.RUNNING)
    except SupersededAttemptError:
        raise
    except Exception as error:
        recording_error = error

    async def clean() -> None:
        async with asyncio.timeout(timeout_seconds) as deadline:
            report = await teardown_run(context, timeout=timeout_seconds)
        if deadline.expired() or report.still_up:
            raise ChildCleanupFailure(invocation.child_instance_id, "child resources remain after teardown")

    cleaning = asyncio.create_task(clean())
    cancelled = False
    try:
        while not cleaning.done():
            try:
                await asyncio.shield(cleaning)
            except asyncio.CancelledError:
                cancelled = True
        cleaning.result()
    except BaseException as error:
        failure = error if isinstance(error, ChildCleanupFailure) else ChildCleanupFailure(
            invocation.child_instance_id, f"teardown failed: {type(error).__name__}",
        )
        try:
            store.set_cleanup_state(invocation, state=ChildCleanupState.FAILED, error=str(failure))
        except Exception as persistence_error:
            failure.add_note(f"Recording failed cleanup also failed: {type(persistence_error).__name__}")
        if recording_error is not None:
            failure.add_note(f"Recording cleanup start also failed: {type(recording_error).__name__}")
        if failure is error:
            raise
        raise failure from error
    try:
        invocation = store.set_cleanup_state(invocation, state=ChildCleanupState.COMPLETED)
    except SupersededAttemptError:
        raise
    except Exception as error:
        raise ChildCleanupFailure(invocation.child_instance_id, "completed teardown could not be recorded") from error
    if cancelled:
        raise asyncio.CancelledError
    return invocation


async def run_child(
    *, step: TaskStep, context: TaskStepContext, task: dict, inputs: dict, iteration: int,
    child_timeout_seconds: float, cleanup_timeout_seconds: float,
) -> TaskStepContext:
    execution = current_step_execution()
    if execution.instance_id != context.instance_id or execution.step_id != step.id:
        raise RuntimeError("Nested task parent does not match the executing step")
    parent = get_task_instance_store().get(execution.instance_id)
    store = get_child_invocation_store()
    initial = _initial_context(
        parent=context, inputs=inputs, execution=execution,
        generation=parent.child_step_generations.get(step.id, 0), iteration=iteration,
    )
    captured_context = initial.to_safe_dict()
    for key in ("workflow_id", "run_group_id"):
        captured_context["metadata"].pop(key, None)
    captured = {
        "context": captured_context, "child_timeout_seconds": child_timeout_seconds,
        "cleanup_timeout_seconds": cleanup_timeout_seconds,
    }
    invocation = store.get_or_create(
        parent_instance_id=execution.instance_id, parent_step_id=step.id,
        parent_step_generation=parent.child_step_generations.get(step.id, 0),
        parent_attempt_epoch=execution.attempt_epoch, iteration=iteration,
        task_definition=task, captured_inputs=captured,
    )
    child_task = Task.from_dict(invocation.task_definition)
    child_context, completed, finished, restart = _restore_context(child_task, invocation, initial)
    cleanup_context = child_context
    cleanup_attempted = False
    primary: BaseException | None = None
    try:
        if not finished:
            if invocation.cleanup_state in (ChildCleanupState.RUNNING, ChildCleanupState.FAILED) or (
                restart and invocation.cleanup_state is not ChildCleanupState.COMPLETED
            ):
                with run_scope(child_task.id):
                    previous = get_task_instance_store().get(invocation.child_instance_id)
                previous_context = TaskStepContext.from_dict(copy.deepcopy(previous.context or {}))
                cleanup_context = previous_context
                cleanup_attempted = True
                invocation = await _clean_child(invocation, previous_context, timeout_seconds=cleanup_timeout_seconds)
                cleanup_context = child_context
                cleanup_attempted = False
            if restart:
                with run_scope(child_task.id):
                    instances = get_task_instance_store()
                    previous = instances.get(invocation.child_instance_id)
                    instances.undo_steps_sync(
                        invocation.child_instance_id, {child_step.id for child_step in child_task.steps},
                        bump_epoch=True, attempt_epoch=previous.attempt_epoch,
                    )
            invocation = store.set_cleanup_state(invocation, state=ChildCleanupState.PENDING)
            try:
                async with asyncio.timeout(child_timeout_seconds):
                    await child_task.run(
                        context=child_context, instance_id=invocation.child_instance_id, completed_step_ids=completed,
                    )
            except TimeoutError as error:
                raise ChildTaskTimeout(invocation.child_instance_id, "execution deadline expired") from error
        status, reason = run_outcome(child_context)
        if status is not RunStatus.COMPLETED:
            raise ChildTaskFailed(invocation.child_instance_id, reason or "child graph failed")
        invocation = store.complete(invocation, outcome=child_context.to_safe_dict())
    except asyncio.CancelledError as error:
        primary = error
        raise
    except BaseException as error:
        primary = error
        if isinstance(error, (ChildTaskFailed, SupersededAttemptError)):
            raise
        raise ChildTaskFailed(invocation.child_instance_id, f"{type(error).__name__}: {error}") from error
    finally:
        if invocation.cleanup_state is not ChildCleanupState.COMPLETED and not cleanup_attempted:
            try:
                await _clean_child(invocation, cleanup_context, timeout_seconds=cleanup_timeout_seconds)
            except BaseException as cleanup_error:
                if primary is not None:
                    cleanup_error.add_note(f"Child execution also failed: {type(primary).__name__}")
                raise
    return child_context
