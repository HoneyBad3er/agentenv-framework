"""Step commits isolate failed work while retaining resources that still need teardown."""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.store import LocalFilesystemObjectStore, LocalSqliteDocumentStore
from agent_env.task import Task
from agent_env.task import store as task_store
from agent_env.task.child_store import get_child_invocation_store
from agent_env.task.execution import ContextWriteConflict
from agent_env.task.nested import ChildTaskFailed
from agent_env.task.persistence import StoreDeadlineExceeded
from agent_env.task.resources import load_resource_context
from agent_env.task.store import TaskStepStatus, get_task_instance_store
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, PromptResponse, TaskStepContext
from agent_env.task_step.context_ops import ContextUpdateOps
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep


TEST_DEADLINE_SECONDS = 10.0
CHILD_TIMEOUT_SECONDS = 5.0
CLEANUP_TIMEOUT_SECONDS = 5.0


@dataclass
class IsolationState:
    calls: Counter[str] = field(default_factory=Counter)
    events: dict[str, asyncio.Event] = field(default_factory=dict)
    observed: list[tuple[str, dict]] = field(default_factory=list)
    after_wait: list[tuple[str, dict]] = field(default_factory=list)
    terminated: list[str] = field(default_factory=list)

    def event(self, name: str) -> asyncio.Event:
        return self.events.setdefault(name, asyncio.Event())


class IsolationStep(TaskStep):
    type = "step_isolation_probe"
    entity_refs = ()
    state = IsolationState()

    def __init__(
        self, *, id: str, version: int | None = None, action: str = "write", writes: dict | None = None,
        peers: list[str] | None = None, wait_commit: str | None = None,
        depends_on: list | None = None, fail_task_on_error: bool = True,
    ) -> None:
        super().__init__(id=id, version=version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.action = action
        self.writes = copy.deepcopy(writes or {})
        self.peers = list(peers or [])
        self.wait_commit = wait_commit

    def to_dict(self) -> dict:
        return {
            **super().to_dict(), "action": self.action, "writes": copy.deepcopy(self.writes),
            "peers": self.peers, "wait_commit": self.wait_commit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IsolationStep:
        return cls(
            **cls._base_from_dict(data), action=data["action"], writes=data["writes"], peers=data["peers"],
            wait_commit=data["wait_commit"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        state = self.state
        state.calls[self.id] += 1
        state.observed.append((self.id, copy.deepcopy(context.metadata)))
        if self.action == "wait_dirty_then_write":
            await state.event("dirty-written").wait()
            state.after_wait.append((self.id, copy.deepcopy(context.metadata)))
        elif self.action in ("parallel_write", "parallel_replace_domain", "parallel_unset"):
            state.event(f"ready:{self.id}").set()
            for peer in self.peers:
                await state.event(f"ready:{peer}").wait()
            if self.wait_commit is not None:
                await state.event(f"committed:{self.wait_commit}").wait()
            if self.action == "parallel_replace_domain":
                context.metadata["domain"] = copy.deepcopy(self.writes["replacement"])
                return context
            if self.action == "parallel_unset":
                context.metadata.pop("domain")
                return context
        elif self.action == "dirty_then_fail":
            context.metadata.setdefault("domain", {})["dirty"] = "failed mutation"
            context.prompt_responses.append(PromptResponse(prompt_id=self.id, response="uncommitted response"))
            context.agent_model = "uncommitted-model"
            state.event("dirty-written").set()
            await state.event("release-failure").wait()
            raise RuntimeError("step failed after changing its private context")
        elif self.action in ("dirty_resource_failure", "dirty_resource_wait"):
            context.metadata.setdefault("domain", {})["dirty"] = "failed mutation"
            context.prompt_responses.append(PromptResponse(prompt_id=self.id, response="uncommitted response"))
            context.agent_model = "uncommitted-model"
            context.deployed_sandboxes.append(DeployedSandbox(
                sandbox_name=self.id, sandbox_id=f"{self.id}-resource", sandbox_mode="container", sandbox_type="isolation-test",
            ))
            state.event("resource-recorded").set()
            if self.action == "dirty_resource_wait":
                await state.event("release-resource").wait()
            raise RuntimeError("step failed after recording a resource")
        elif self.action in ("checkpoint_then_enrich", "checkpoint_then_fail"):
            agent = DeployedAgent(
                agent_name=self.id, api_url="http://agent.test", sandbox_id="checkpointed-agent", sandbox_type="isolation-test",
            )
            context.deployed_agents.append(agent)
            await context.checkpoint_resources()
            state.event("resource-checkpointed").set()
            await state.event("release-checkpointed-resource").wait()
            if self.action == "checkpoint_then_fail":
                context.metadata.setdefault("domain", {})["dirty"] = "uncommitted"
                raise RuntimeError("step failed after its temporary resource checkpoint")
            agent.a2a_card = {"name": "enriched-agent", "capabilities": {"streaming": True}}
        elif self.action == "release_failure":
            state.event("release-failure").set()
        elif self.action == "increment_then_fail_once":
            context.metadata["count"] += 1
            if state.calls[self.id] == 1:
                context.metadata["failed_only"] = "uncommitted"
                state.event("dirty-written").set()
                await state.event("release-failure").wait()
                raise RuntimeError("parallel child failed after changing its private context")
        elif self.action == "copy_count":
            context.metadata["answer"] = context.metadata["count"]
        elif self.action == "read_upstream":
            context.metadata["downstream"] = copy.deepcopy(context.metadata["domain"]["upstream"])
        for key, value in self.writes.items():
            context.metadata.setdefault("domain", {})[key] = copy.deepcopy(value)
        return context


class IsolationProvider(SandboxProvider):
    state = IsolationState()

    async def create_sandbox(self, **parameters) -> Sandbox:
        raise AssertionError("the isolation fixture never provisions a real sandbox")

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        raise AssertionError("native teardown must terminate the recorded sandbox by id")

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        self.state.terminated.append(sandbox_id)


@pytest.fixture
def isolation_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IsolationState:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[task_steps]\nimpls = ["tst.unit.task.step_isolation_test:IsolationStep"]\n'
        '[sandbox.providers.isolation-test]\nimpl = "tst.unit.task.step_isolation_test:IsolationProvider"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config_path))
    configure(
        document_store=LocalSqliteDocumentStore(path=str(tmp_path / "documents.sqlite3")),
        object_store=LocalFilesystemObjectStore(root=str(tmp_path / "objects")),
    )
    state = IsolationState()
    monkeypatch.setattr(IsolationStep, "state", state)
    monkeypatch.setattr(IsolationProvider, "state", state)
    yield state
    reset_config()


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal", (False, True))
async def test_successful_sibling_never_publishes_or_journals_another_steps_failed_mutations(
    isolation_state: IsolationState, fatal: bool,
) -> None:
    task = Task(id="failed-sibling-isolation", version=1, steps=[
        IsolationStep(id="successful", action="wait_dirty_then_write", writes={"committed": "kept"}, depends_on=[]),
        IsolationStep(id="failed", action="dirty_then_fail", depends_on=[], fail_task_on_error=fatal),
        IsolationStep(id="release", action="release_failure", depends_on=["successful"]),
    ])
    context = TaskStepContext(metadata={"domain": {}})

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        if fatal:
            with pytest.raises(RuntimeError, match="step failed after changing its private context"):
                await task.run(context=context)
        else:
            await task.run(context=context)

    assert context.metadata["domain"] == {"committed": "kept"}
    assert context.prompt_responses == [] and context.agent_model is None
    assert len(isolation_state.after_wait) == 1
    assert isolation_state.after_wait[0][0] == "successful"
    assert isolation_state.after_wait[0][1]["domain"] == {}
    instance = get_task_instance_store().get(context.instance_id)
    assert instance.context["metadata"]["domain"] == {"committed": "kept"}
    journal = get_task_instance_store().journal_entries_sync(context.instance_id)
    successful_entry = next(entry for entry in journal if entry["step_id"] == "successful")
    changes = ContextUpdateOps.from_journal_dict(successful_entry["ops"])
    assert not any("dirty" in path or "prompt_responses" in path or "agent_model" in path for path in changes.sets)
    replayed = get_task_instance_store().replay_completed_context_sync(context.instance_id, {"successful"})
    assert replayed["metadata"]["domain"] == {"committed": "kept"}
    assert replayed["prompt_responses"] == [] and replayed["agent_model"] is None


@pytest.mark.asyncio
async def test_disjoint_parallel_changes_to_nested_metadata_merge_without_losing_either_commit(
    isolation_state: IsolationState,
) -> None:
    task = Task(id="disjoint-parallel-commits", version=1, steps=[
        IsolationStep(id="left", action="parallel_write", writes={"left": {"value": 1}}, peers=["right"], depends_on=[]),
        IsolationStep(id="right", action="parallel_write", writes={"right": {"value": 2}}, peers=["left"], depends_on=[]),
    ])
    context = TaskStepContext(metadata={"domain": {}})

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        result = await task.run(context=context)

    expected = {"left": {"value": 1}, "right": {"value": 2}}
    assert result.metadata["domain"] == expected
    assert isolation_state.calls == {"left": 1, "right": 1}
    instance = get_task_instance_store().get(result.instance_id)
    assert instance.status == "completed"
    assert instance.context["metadata"]["domain"] == expected
    replayed = get_task_instance_store().replay_completed_context_sync(result.instance_id, {"left", "right"})
    assert replayed["metadata"]["domain"] == expected


@pytest.mark.asyncio
async def test_dependent_step_starts_with_the_committed_upstream_result(isolation_state: IsolationState) -> None:
    task = Task(id="dependent-committed-context", version=1, steps=[
        IsolationStep(id="upstream", writes={"upstream": {"value": "committed"}}),
        IsolationStep(id="downstream", action="read_upstream", depends_on=["upstream"]),
    ])

    callbacks: list[tuple[str, dict]] = []

    def observe_completion(
        index: int, total: int, step: TaskStep, context: TaskStepContext, duration: float,
    ) -> None:
        callbacks.append((step.id, copy.deepcopy(context.metadata)))

    result = await task.run(context=TaskStepContext(metadata={"domain": {}}), on_step_complete=observe_completion)

    assert result.metadata["downstream"] == {"value": "committed"}
    observed = next(metadata for step, metadata in isolation_state.observed if step == "downstream")
    assert observed["domain"]["upstream"] == {"value": "committed"}
    assert callbacks[0][1]["domain"]["upstream"] == {"value": "committed"}
    assert callbacks[1][1]["downstream"] == {"value": "committed"}
    assert get_task_instance_store().get(result.instance_id).context["metadata"]["downstream"] == {"value": "committed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ("fatal", "tolerant", "cancelled"))
async def test_failed_or_cancelled_step_retains_cleanup_resources_without_publishing_partial_domain_values(
    isolation_state: IsolationState, ending: str,
) -> None:
    action = "dirty_resource_wait" if ending == "cancelled" else "dirty_resource_failure"
    task = Task(id="failed-resource-isolation", version=1, steps=[IsolationStep(
        id="resource-owner", action=action, fail_task_on_error=ending != "tolerant",
    )])
    context = TaskStepContext(metadata={"domain": {"seed": "kept"}, "retained": "parent value"})
    if ending == "cancelled":
        running = asyncio.create_task(task.run(context=context))
        try:
            async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                await isolation_state.event("resource-recorded").wait()
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(TEST_DEADLINE_SECONDS):
                    await running
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    elif ending == "fatal":
        with pytest.raises(RuntimeError, match="step failed after recording a resource"):
            await task.run(context=context)
    else:
        await task.run(context=context)

    assert context.metadata["domain"] == {"seed": "kept"}
    assert context.metadata["retained"] == "parent value"
    assert context.prompt_responses == [] and context.agent_model is None
    assert [sandbox.sandbox_id for sandbox in context.deployed_sandboxes] == ["resource-owner-resource"]
    instance = get_task_instance_store().get(context.instance_id)
    resources = await load_resource_context(
        context.instance_id, deadline=asyncio.get_running_loop().time() + CLEANUP_TIMEOUT_SECONDS,
    )
    assert [sandbox.sandbox_id for sandbox in resources.deployed_sandboxes] == ["resource-owner-resource"]
    assert instance.context["metadata"]["domain"] == {"seed": "kept"}
    report = await teardown_run(context, timeout=CLEANUP_TIMEOUT_SECONDS)
    assert report.still_up == ()
    assert isolation_state.terminated == ["resource-owner-resource"]


@pytest.mark.asyncio
async def test_child_resume_after_a_failed_parallel_branch_reuses_successful_clean_commits(
    isolation_state: IsolationState,
) -> None:
    child = Task(id="parallel-child", version=1, steps=[
        IsolationStep(id="successful", action="wait_dirty_then_write", writes={"committed": "kept"}, depends_on=[]),
        IsolationStep(id="failed-once", action="increment_then_fail_once", depends_on=[]),
        IsolationStep(id="release", action="release_failure", depends_on=["successful"]),
        IsolationStep(id="join", action="copy_count", depends_on=["failed-once", "release"]),
    ])
    task = Task(id="parallel-child-parent", version=1, steps=[SubDAGTaskStep(
        id="nested", version=None, task=child, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
        cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS, input_bindings={"count": "initial"}, output_bindings={"result": "answer"},
    )])
    context = TaskStepContext(metadata={"initial": 0})
    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        with pytest.raises(ChildTaskFailed):
            await task.run(context=context)
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    previous = get_task_instance_store().get(context.instance_id)
    completed = {step.step_id for step in previous.completed_steps if step.status is TaskStepStatus.SUCCESS}

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        resumed = await task.run(
            context=TaskStepContext.from_dict(previous.context), instance_id=context.instance_id, completed_step_ids=completed,
        )

    assert resumed.metadata["result"] == 1
    assert isolation_state.calls["successful"] == 1
    assert isolation_state.calls["failed-once"] == 2
    assert "failed_only" not in resumed.metadata
    child_instance = get_task_instance_store().get(invocation.child_instance_id)
    assert child_instance.context["metadata"]["domain"] == {"committed": "kept"}
    assert child_instance.context["metadata"]["count"] == 1
    assert "failed_only" not in child_instance.context["metadata"]
    observed = [metadata["count"] for step, metadata in isolation_state.observed if step == "failed-once"]
    assert observed == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ("same_leaf", "ancestor", "unset_ancestor"))
@pytest.mark.parametrize("winning_step", ("left", "right"))
async def test_conflicting_parallel_writes_are_fatal_and_never_publish_a_partial_losing_commit(
    isolation_state: IsolationState, conflict: str, winning_step: str,
) -> None:
    left_action = "parallel_write"
    left_writes = {"shared": "left", "left_only": "left"}
    right_writes = {"shared": "right", "right_only": "right"}
    if conflict == "ancestor":
        left_action = "parallel_replace_domain"
        left_writes = {"replacement": "left scalar"}
        right_writes = {"right_only": "right"}
    elif conflict == "unset_ancestor":
        left_action = "parallel_unset"
        left_writes = {}
        right_writes = {"right_only": "right"}
    task = Task(id="conflicting-parallel-commits", version=1, steps=[
        IsolationStep(
            id="left", action=left_action, writes=left_writes, peers=["right"], depends_on=[], fail_task_on_error=False,
            wait_commit="right" if winning_step == "right" else None,
        ),
        IsolationStep(
            id="right", action="parallel_write", writes=right_writes, peers=["left"], depends_on=[], fail_task_on_error=False,
            wait_commit="left" if winning_step == "left" else None,
        ),
    ])
    context = TaskStepContext(metadata={"domain": {"original": "seed"}})

    def release_after_commit(
        index: int, total: int, step: TaskStep, context: TaskStepContext, duration: float,
    ) -> None:
        isolation_state.event(f"committed:{step.id}").set()

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        with pytest.raises(ContextWriteConflict):
            await task.run(context=context, on_step_complete=release_after_commit)

    instance = get_task_instance_store().get(context.instance_id)
    winners = {step.step_id for step in instance.completed_steps if step.status is TaskStepStatus.SUCCESS}
    assert len(winners) == 1
    winner = next(iter(winners))
    assert winner == winning_step
    replayed = get_task_instance_store().replay_completed_context_sync(context.instance_id, winners)
    if winner == "left" and conflict == "unset_ancestor":
        assert "domain" not in context.metadata
        assert "domain" not in instance.context["metadata"]
        assert "domain" not in replayed["metadata"]
    else:
        if winner == "left" and conflict == "ancestor":
            expected = "left scalar"
        else:
            expected = {"original": "seed", **(left_writes if winner == "left" else right_writes)}
        assert context.metadata["domain"] == expected
        assert instance.context["metadata"]["domain"] == expected
        assert replayed["metadata"]["domain"] == expected
    assert instance.status == "failed"
    assert context.metadata["failed_steps"][-1]["error_type"] == "ContextWriteConflict"


@pytest.mark.asyncio
async def test_identical_parallel_writes_are_idempotent_and_keep_disjoint_companion_values(
    isolation_state: IsolationState,
) -> None:
    task = Task(id="identical-parallel-commits", version=1, steps=[
        IsolationStep(id="left", action="parallel_write", writes={"shared": 1, "left": 2}, peers=["right"], depends_on=[]),
        IsolationStep(id="right", action="parallel_write", writes={"shared": 1, "right": 3}, peers=["left"], depends_on=[]),
    ])

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        result = await task.run(context=TaskStepContext(metadata={"domain": {}}))

    assert result.metadata["domain"] == {"shared": 1, "left": 2, "right": 3}
    assert isolation_state.calls == {"left": 1, "right": 1}
    instance = get_task_instance_store().get(result.instance_id)
    assert instance.status == "completed"
    assert instance.context["metadata"]["domain"] == {"shared": 1, "left": 2, "right": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("fails_after_checkpoint", (False, True))
async def test_early_resource_checkpoint_is_private_until_commit_and_enrichment_does_not_duplicate_identity(
    isolation_state: IsolationState, fails_after_checkpoint: bool,
) -> None:
    task = Task(id="checkpoint-resource-enrichment", version=1, steps=[IsolationStep(
        id="agent-owner", action="checkpoint_then_fail" if fails_after_checkpoint else "checkpoint_then_enrich",
    )])
    context = TaskStepContext(metadata={"domain": {"seed": "kept"}})
    running = asyncio.create_task(task.run(context=context))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await isolation_state.event("resource-checkpointed").wait()
        assert context.deployed_agents == []
        early_resources = await load_resource_context(
            context.instance_id, deadline=asyncio.get_running_loop().time() + CLEANUP_TIMEOUT_SECONDS,
        )
        assert len(early_resources.deployed_agents) == 1
        assert early_resources.deployed_agents[0].sandbox_id == "checkpointed-agent"
        assert early_resources.deployed_agents[0].a2a_card is None
        isolation_state.event("release-checkpointed-resource").set()
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            if fails_after_checkpoint:
                with pytest.raises(RuntimeError, match="step failed after its temporary resource checkpoint"):
                    await running
            else:
                await running
    finally:
        isolation_state.event("release-checkpointed-resource").set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    expected_card = None if fails_after_checkpoint else {"name": "enriched-agent", "capabilities": {"streaming": True}}
    assert len(context.deployed_agents) == 1
    assert context.deployed_agents[0].sandbox_id == "checkpointed-agent"
    assert context.deployed_agents[0].a2a_card == expected_card
    assert context.metadata["domain"] == {"seed": "kept"}
    resources = await load_resource_context(
        context.instance_id, deadline=asyncio.get_running_loop().time() + CLEANUP_TIMEOUT_SECONDS,
    )
    assert len(resources.deployed_agents) == 1
    assert resources.deployed_agents[0].a2a_card == expected_card
    report = await teardown_run(context, timeout=CLEANUP_TIMEOUT_SECONDS)
    assert report.still_up == ()
    assert isolation_state.terminated == ["checkpointed-agent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tolerant", (False, True))
async def test_completion_deadline_failure_never_publishes_unacknowledged_step_outputs(
    isolation_state: IsolationState, monkeypatch: pytest.MonkeyPatch, tolerant: bool,
) -> None:
    async def fail_completion(
        instance_id: str, step_id: str, ops: ContextUpdateOps, total_steps: int, completed_at_utc: str,
        status: TaskStepStatus = TaskStepStatus.SUCCESS, attempt_epoch: int = 0,
    ) -> None:
        raise StoreDeadlineExceeded("step completion write exceeded its deadline")

    monkeypatch.setattr(task_store, "record_step_complete", fail_completion)
    task = Task(id="unacknowledged-commit", version=1, steps=[IsolationStep(
        id="producer", writes={"unacknowledged": "must not publish"}, fail_task_on_error=not tolerant,
    )])
    context = TaskStepContext(metadata={"domain": {"seed": "kept"}})

    with pytest.raises(StoreDeadlineExceeded):
        await task.run(context=context)

    assert context.metadata["domain"] == {"seed": "kept"}
    instance = get_task_instance_store().get(context.instance_id)
    assert instance.status == "failed"
    assert instance.context["metadata"]["domain"] == {"seed": "kept"}
    assert not any(step.status is TaskStepStatus.SUCCESS for step in instance.completed_steps)
    replayed = get_task_instance_store().replay_completed_context_sync(context.instance_id, set())
    assert replayed["metadata"]["domain"] == {"seed": "kept"}


@pytest.mark.asyncio
async def test_blocked_completion_ack_keeps_outputs_out_of_caller_context_and_store_until_it_returns(
    isolation_state: IsolationState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_entered = asyncio.Event()
    completion_release = asyncio.Event()
    complete = task_store.record_step_complete
    callbacks: list[str] = []

    async def block_completion(
        instance_id: str, step_id: str, ops: ContextUpdateOps, total_steps: int, completed_at_utc: str,
        status: TaskStepStatus = TaskStepStatus.SUCCESS, attempt_epoch: int = 0,
    ) -> None:
        completion_entered.set()
        await completion_release.wait()
        await complete(
            instance_id=instance_id, step_id=step_id, ops=ops, total_steps=total_steps,
            completed_at_utc=completed_at_utc, status=status, attempt_epoch=attempt_epoch,
        )

    def observe_completion(index: int, total: int, step: TaskStep, context: TaskStepContext, duration: float) -> None:
        callbacks.append(step.id)

    monkeypatch.setattr(task_store, "record_step_complete", block_completion)
    task = Task(id="pending-commit", version=1, steps=[IsolationStep(id="producer", writes={"committed": "acknowledged"})])
    context = TaskStepContext(metadata={"domain": {"seed": "kept"}})
    running = asyncio.create_task(task.run(context=context, on_step_complete=observe_completion))
    try:
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await completion_entered.wait()
        assert context.metadata["domain"] == {"seed": "kept"}
        assert get_task_instance_store().get(context.instance_id).context["metadata"]["domain"] == {"seed": "kept"}
        assert callbacks == []
        completion_release.set()
        async with asyncio.timeout(TEST_DEADLINE_SECONDS):
            await running
    finally:
        completion_release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    expected = {"seed": "kept", "committed": "acknowledged"}
    assert context.metadata["domain"] == expected
    assert get_task_instance_store().get(context.instance_id).context["metadata"]["domain"] == expected
    assert callbacks == ["producer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("left_key", "right_key"), (("left.value", "right.value"), ("$left", "$right")))
async def test_parallel_maps_with_non_addressable_keys_conflict_instead_of_recording_silent_sibling_loss(
    isolation_state: IsolationState, left_key: str, right_key: str,
) -> None:
    proposed = {"left": {left_key: "left"}, "right": {right_key: "right"}}
    task = Task(id="atomic-dictionary-commits", version=1, steps=[
        IsolationStep(id="left", action="parallel_write", writes=proposed["left"], peers=["right"], depends_on=[]),
        IsolationStep(id="right", action="parallel_write", writes=proposed["right"], peers=["left"], depends_on=[]),
    ])
    context = TaskStepContext(metadata={"domain": {}})

    async with asyncio.timeout(TEST_DEADLINE_SECONDS):
        with pytest.raises(ContextWriteConflict):
            await task.run(context=context)

    instance = get_task_instance_store().get(context.instance_id)
    winners = [step.step_id for step in instance.completed_steps if step.status is TaskStepStatus.SUCCESS]
    assert len(winners) <= 1
    expected = proposed[winners[0]] if winners else {}
    assert context.metadata["domain"] == expected
    assert instance.context["metadata"]["domain"] == expected
    assert instance.status == "failed"
