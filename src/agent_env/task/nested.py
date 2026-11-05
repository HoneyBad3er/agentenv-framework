"""Execute isolated child graphs under durable parent step identities."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from functools import partial
from typing import NoReturn, TypeVar

from agent_env.runner.runner import RunStatus, run_outcome
from agent_env.store.base import NotFoundError
from agent_env.store.routing import run_scope
from agent_env.task.child_store import ChildCleanupState, ChildInvocation, get_child_invocation_store
from agent_env.task.execution import StepExecution, TaskLifecycleError, TaskPersistenceFailure, current_step_execution
from agent_env.task.persistence import (
    call_store, current_store_deadline, is_task_persistence_required, raise_persistence_failure, store_deadline,
)
from agent_env.task.recovery import merge_resource_contexts, pending_resource_context
from agent_env.task.resources import load_resource_context
from agent_env.task.store import SupersededAttemptError, TaskStepStatus, get_task_instance_store
from agent_env.task.task import Task
from agent_env.task.teardown import recorded_sandboxes, teardown_run
from agent_env.task_step.context import HEARTBEAT_KEY, TaskStepContext, regraft_redacted_keys
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY


class ChildTaskFailed(RuntimeError):
    """A child graph did not produce a successful outcome."""

    def __init__(self, child_instance_id: str, reason: str) -> None:
        self.child_instance_id = child_instance_id
        super().__init__(f"Child task {child_instance_id}: {reason}")


class ChildTaskTimeout(ChildTaskFailed):
    """The configured child execution deadline expired."""


class ChildCleanupFailure(ChildTaskFailed, TaskLifecycleError):
    """Child resources could not be confirmed terminated."""


_T = TypeVar("_T")


async def _call_nested_store(operation: Callable[[], _T], *, deadline: float) -> _T:
    try:
        return await call_store(operation, deadline=deadline)
    except (SupersededAttemptError, TaskLifecycleError):
        raise
    except Exception as error:
        raise_persistence_failure(error, operation="persist nested task invocation")
        raise


def _initial_context(
    *, parent: TaskStepContext, inputs: dict, execution: StepExecution, generation: int, iteration: int,
    invocation_name: str | None,
) -> TaskStepContext:
    metadata = copy.deepcopy(inputs)
    metadata["user_overrides"] = copy.deepcopy(parent.metadata.get("user_overrides") or {})
    metadata["user_overrides"].pop("step_params", None)
    metadata["parent_task"] = {
        "instance_id": execution.instance_id,
        "step_id": execution.step_id,
        "step_generation": generation,
        "iteration": iteration,
        "invocation_name": invocation_name,
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


async def _restore_context(
    task: Task, invocation: ChildInvocation, initial: TaskStepContext, *, deadline: float,
) -> tuple[TaskStepContext, set[str], bool, bool]:
    if invocation.outcome is not None:
        return TaskStepContext.from_dict(copy.deepcopy(invocation.outcome)), set(), True, False
    try:
        with run_scope(task.id):
            instance = await call_store(partial(get_task_instance_store().get, invocation.child_instance_id), deadline=deadline)
    except NotFoundError:
        return initial, set(), False, False
    except (SupersededAttemptError, TaskLifecycleError):
        raise
    except Exception as error:
        raise_persistence_failure(error, operation="restore child task instance")
        raise
    if instance.task_id != task.id or instance.task_version != task.version:
        raise ChildTaskFailed(invocation.child_instance_id, "stored task identity changed")
    restored = copy.deepcopy(instance.context or {})
    regraft_redacted_keys(initial.to_dict(), restored)
    context = TaskStepContext.from_dict(restored)
    completed = {step.step_id for step in instance.completed_steps if step.status is TaskStepStatus.SUCCESS}
    status, _ = run_outcome(context)
    if instance.status == "completed" and status is RunStatus.COMPLETED:
        return context, completed, True, False
    inventory = await load_resource_context(invocation.child_instance_id, deadline=deadline)
    captured = TaskStepContext.from_dict(copy.deepcopy(invocation.cleanup_context or {}))
    if recorded_sandboxes(merge_resource_contexts(context, inventory, captured)):
        return initial, set(), False, True
    retained = _completed_dependencies(task, completed)
    with run_scope(task.id):
        restored = await _call_nested_store(partial(
            get_task_instance_store().replay_completed_context_sync, invocation.child_instance_id, retained,
        ), deadline=deadline)
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
    context = pending_resource_context(merge_resource_contexts(
        context, TaskStepContext.from_dict(invocation.cleanup_context or {}),
    ))
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    recording_error: Exception | None = None
    try:
        invocation = await _call_nested_store(partial(
            store.capture_cleanup_context, invocation, context=context.to_safe_dict(),
        ), deadline=deadline)
        invocation = await _call_nested_store(partial(
            store.set_cleanup_state, invocation, state=ChildCleanupState.RUNNING,
        ), deadline=deadline)
    except SupersededAttemptError:
        raise
    except Exception as error:
        recording_error = error

    def raise_cleanup_cancellation(
        cancellation: asyncio.CancelledError, *, cleanup_failure: ChildCleanupFailure | None = None,
    ) -> NoReturn:
        if is_task_persistence_required() and recording_error is not None:
            failures = ((cancellation, recording_error) if cleanup_failure is None
                        else (cancellation, recording_error, cleanup_failure))
            raise TaskPersistenceFailure("Child cleanup was cancelled after required persistence failed") from BaseExceptionGroup(
                "Child cleanup cancellation and persistence failure", failures,
            )
        raise cancellation

    async def clean() -> None:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ChildCleanupFailure(invocation.child_instance_id, "cleanup persistence exhausted its deadline")
        async with asyncio.timeout(remaining) as cleanup_deadline:
            report = await teardown_run(context, timeout=remaining)
        if cleanup_deadline.expired() or report.still_up:
            raise ChildCleanupFailure(invocation.child_instance_id, "child resources remain after teardown")
        context.metadata[TORN_DOWN_KEY] = sorted(
            set(context.metadata.get(TORN_DOWN_KEY) or ()) | {resource.sandbox_id for resource in recorded_sandboxes(context)},
        )

    cleaning = asyncio.create_task(clean())
    cancellation: asyncio.CancelledError | None = None
    try:
        while not cleaning.done():
            try:
                await asyncio.shield(cleaning)
            except asyncio.CancelledError as error:
                cancellation = error
        cleaning.result()
    except BaseException as error:
        failure = error if isinstance(error, ChildCleanupFailure) else ChildCleanupFailure(
            invocation.child_instance_id, f"teardown failed: {type(error).__name__}",
        )
        try:
            report_deadline = asyncio.get_running_loop().time() + timeout_seconds
            invocation = await _call_nested_store(partial(
                store.capture_cleanup_context, invocation, context=context.to_safe_dict(),
            ), deadline=report_deadline)
            await _call_nested_store(partial(
                store.set_cleanup_state, invocation, state=ChildCleanupState.FAILED, error=str(failure),
            ), deadline=report_deadline)
        except asyncio.CancelledError as cancellation:
            raise_cleanup_cancellation(cancellation, cleanup_failure=failure)
        except Exception as persistence_error:
            failure.add_note(f"Recording failed cleanup also failed: {type(persistence_error).__name__}")
        if recording_error is not None:
            failure.add_note(f"Recording cleanup start also failed: {type(recording_error).__name__}")
        if failure is error:
            raise
        raise failure from error
    try:
        report_deadline = asyncio.get_running_loop().time() + timeout_seconds
        invocation = await _call_nested_store(partial(
            store.capture_cleanup_context, invocation, context=context.to_safe_dict(),
        ), deadline=report_deadline)
        invocation = await _call_nested_store(partial(
            store.set_cleanup_state, invocation, state=ChildCleanupState.COMPLETED,
        ), deadline=report_deadline)
    except SupersededAttemptError:
        raise
    except asyncio.CancelledError as error:
        raise_cleanup_cancellation(error)
    except Exception as error:
        raise ChildCleanupFailure(invocation.child_instance_id, "completed teardown could not be recorded") from error
    if cancellation is not None:
        raise_cleanup_cancellation(cancellation)
    if is_task_persistence_required() and recording_error is not None:
        raise recording_error
    return invocation


async def run_child(
    *, step: TaskStep, context: TaskStepContext, task: dict, inputs: dict, iteration: int,
    child_timeout_seconds: float, cleanup_timeout_seconds: float,
    invocation_name: str | None = None,
) -> TaskStepContext:
    execution = current_step_execution()
    if execution.instance_id != context.instance_id or execution.step_id != step.id:
        raise RuntimeError("Nested task parent does not match the executing step")
    deadline = asyncio.get_running_loop().time() + child_timeout_seconds
    parent = await _call_nested_store(partial(get_task_instance_store().get, execution.instance_id), deadline=deadline)
    store = get_child_invocation_store()
    initial = _initial_context(
        parent=context, inputs=inputs, execution=execution,
        generation=parent.child_step_generations.get(step.id, 0), iteration=iteration,
        invocation_name=invocation_name,
    )
    captured_context = initial.to_safe_dict()
    for key in ("workflow_id", "run_group_id"):
        captured_context["metadata"].pop(key, None)
    captured = {
        "context": captured_context, "child_timeout_seconds": child_timeout_seconds,
        "cleanup_timeout_seconds": cleanup_timeout_seconds,
    }
    invocation = await _call_nested_store(partial(store.get_or_create,
        parent_instance_id=execution.instance_id, parent_step_id=step.id,
        parent_step_generation=parent.child_step_generations.get(step.id, 0),
        parent_attempt_epoch=execution.attempt_epoch, iteration=iteration,
        task_definition=task, captured_inputs=captured,
        invocation_name=invocation_name,
    ), deadline=deadline)
    child_task = Task.from_dict(invocation.task_definition)
    child_context = initial
    cleanup_context = merge_resource_contexts(initial, TaskStepContext.from_dict(invocation.cleanup_context or {}))
    cleanup_attempted = False
    primary: BaseException | None = None

    async def checkpoint_resources(resource_context: TaskStepContext) -> None:
        nonlocal invocation, cleanup_context
        cleanup_context = merge_resource_contexts(cleanup_context, resource_context)
        invocation = await _call_nested_store(partial(
            store.capture_cleanup_context, invocation, context=cleanup_context.to_safe_dict(),
        ), deadline=current_store_deadline())

    try:
        child_context, completed, finished, restart = await _restore_context(
            child_task, invocation, initial, deadline=deadline,
        )
        cleanup_context = merge_resource_contexts(cleanup_context, child_context)
        if not finished:
            if invocation.cleanup_state in (ChildCleanupState.RUNNING, ChildCleanupState.FAILED) or (
                restart and invocation.cleanup_state is not ChildCleanupState.COMPLETED
            ):
                with run_scope(child_task.id):
                    previous = await _call_nested_store(partial(
                        get_task_instance_store().get, invocation.child_instance_id,
                    ), deadline=deadline)
                previous_context = TaskStepContext.from_dict(copy.deepcopy(previous.context or {}))
                cleanup_context = merge_resource_contexts(cleanup_context, previous_context)
                cleanup_attempted = True
                invocation = await _clean_child(invocation, cleanup_context, timeout_seconds=cleanup_timeout_seconds)
                cleanup_context = child_context
                cleanup_attempted = False
            if restart:
                with run_scope(child_task.id):
                    instances = get_task_instance_store()
                    previous = await _call_nested_store(partial(instances.get, invocation.child_instance_id), deadline=deadline)
                    await _call_nested_store(partial(instances.undo_steps_sync,
                        invocation.child_instance_id, {child_step.id for child_step in child_task.steps},
                        bump_epoch=True, attempt_epoch=previous.attempt_epoch,
                    ), deadline=deadline)
            invocation = await _call_nested_store(partial(
                store.set_cleanup_state, invocation, state=ChildCleanupState.PENDING,
            ), deadline=deadline)
            try:
                with store_deadline(deadline):
                    async with asyncio.timeout_at(deadline):
                        await child_task.run(
                            context=child_context, instance_id=invocation.child_instance_id, completed_step_ids=completed,
                            resource_checkpoint=checkpoint_resources,
                            resource_checkpoint_timeout_seconds=cleanup_timeout_seconds,
                        )
            except TaskLifecycleError:
                raise
            except TimeoutError as error:
                raise ChildTaskTimeout(invocation.child_instance_id, "execution deadline expired") from error
        status, reason = run_outcome(child_context)
        if status is not RunStatus.COMPLETED:
            raise ChildTaskFailed(invocation.child_instance_id, reason or "child graph failed")
        invocation = await _call_nested_store(
            partial(store.complete, invocation, outcome=child_context.to_safe_dict()), deadline=deadline,
        )
    except asyncio.CancelledError as error:
        primary = error
        raise
    except BaseException as error:
        primary = error
        if isinstance(error, (ChildTaskFailed, SupersededAttemptError, TaskLifecycleError)):
            raise
        raise ChildTaskFailed(invocation.child_instance_id, f"{type(error).__name__}: {error}") from error
    finally:
        if invocation.cleanup_state is not ChildCleanupState.COMPLETED and not cleanup_attempted:
            try:
                cleanup_context = merge_resource_contexts(cleanup_context, child_context)
                await _clean_child(invocation, cleanup_context, timeout_seconds=cleanup_timeout_seconds)
            except BaseException as cleanup_error:
                if primary is not None:
                    cleanup_error.add_note(f"Child execution also failed: {type(primary).__name__}")
                raise
    return child_context
