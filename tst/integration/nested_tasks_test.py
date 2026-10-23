"""Nested graphs own their real local containers through success, failure and cancellation, without a model."""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_env.artifact import DockerImageArtifact, FileArtifact
from agent_env.config import configure
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.task import Task
from agent_env.task.child_store import ChildCleanupState, get_child_invocation_store
from agent_env.task.nested import ChildTaskFailed
from agent_env.task.store import get_task_instance_store
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.repeat_until import RepeatUntilTaskStep
from agent_env.task_step.task_steps.run_code import RunCodeTaskStep
from agent_env.task_step.task_steps.subdag import SubDAGTaskStep

pytestmark = pytest.mark.integration

DOCKER_COMMAND_TIMEOUT_SECONDS = 60
CHILD_TIMEOUT_SECONDS = 90
CLEANUP_TIMEOUT_SECONDS = 30
READINESS_TIMEOUT_SECONDS = 45
TEST_TIMEOUT_SECONDS = 180
SCRIPT_BLOCK_SECONDS = 120
CONTAINER_PORT = 8080
FIXTURE_LABEL = "dev.agentenv.nested-task-fixture"

COUNTER_SCRIPT = """def run(input):
    return {"count": input["results"]["counter"]["count"] + 1}
"""
FAILURE_SCRIPT = """def run(input):
    raise RuntimeError("fixture child failure")
"""
BLOCKING_SCRIPT = f"""import pathlib
import threading

def run(input):
    pathlib.Path('/tmp/nested-child-ready').write_text('ready')
    threading.Event().wait({SCRIPT_BLOCK_SECONDS})
    return {{"unexpected": "completed"}}
"""


def docker_output(*arguments: str) -> str:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, check=True,
                          timeout=DOCKER_COMMAND_TIMEOUT_SECONDS).stdout.strip()


@dataclass(frozen=True)
class DockerFixture:
    image: str
    token: str
    sandbox_root: Path

    def container_ids(self) -> list[str]:
        return docker_output("ps", "-aq", "--filter", f"label={FIXTURE_LABEL}={self.token}").split()


@pytest.fixture
def docker_fixture(tmp_path, monkeypatch):
    docker_output("info", "--format", "{{.ServerVersion}}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandbox_root))
    configure()
    token = uuid.uuid4().hex
    directory = tmp_path / "image"
    directory.mkdir()
    (directory / "Dockerfile").write_text(
        f'FROM python:3.12-slim\nLABEL {FIXTURE_LABEL}={token}\n'
        'CMD ["python3", "-c", "import signal; signal.pause()"]\n',
    )
    image = DockerImageArtifact.put_context(id=f"nested-fixture-image-{token}", description="nested task fixture",
                                           context_path=str(directory), dockerfile_path="Dockerfile")
    provider = LocalSandboxProvider()

    async def prepare() -> None:
        try:
            await provider.prepare_image(image)
        finally:
            await provider.close()

    asyncio.run(prepare())
    fixture = DockerFixture(image=image.image_name, token=token, sandbox_root=sandbox_root)
    try:
        yield fixture
    finally:
        remaining = fixture.container_ids()
        if remaining:
            docker_output("rm", "-f", *remaining)
        docker_output("image", "rm", "-f", fixture.image)


def child_task(fixture: DockerFixture, *, script: str) -> Task:
    artifact = FileArtifact.put_bytes(id=f"nested-script-{uuid.uuid4().hex}", description="child program",
                                      filename="run.py", content=script.encode())
    return Task(id=f"child-{uuid.uuid4().hex}", version=1, steps=[
        DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="child", sandbox_mode="container",
                              sandbox_type="local", image=fixture.image, port=CONTAINER_PORT, memory_mb=512),
        RunCodeTaskStep(id="counter", version=None, sandbox_name="child", script_artifact_id=artifact.id,
                        script_artifact_version=artifact.version, timeout_seconds=CHILD_TIMEOUT_SECONDS),
    ])


def subdag_step(child: Task) -> SubDAGTaskStep:
    return SubDAGTaskStep(id="nested", version=None, task=child, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
                         cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS,
                         input_bindings={"script_results": "initial"}, output_bindings={"answer": "script_results"})


def assert_children_cleaned(fixture: DockerFixture, context: TaskStepContext, *, count: int) -> None:
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    assert len(invocations) == count
    assert all(invocation.cleanup_state is ChildCleanupState.COMPLETED for invocation in invocations)
    assert fixture.container_ids() == []
    assert not tuple(fixture.sandbox_root.glob("agent-env-*"))
    assert context.deployed_sandboxes == []


@pytest.mark.asyncio
async def test_subdag_copies_outputs_after_native_run_code_and_removes_its_container(docker_fixture):
    parent = Task(id="nested-success-parent", version=1, steps=[subdag_step(child_task(docker_fixture, script=COUNTER_SCRIPT))])

    async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
        context = await parent.run(context=TaskStepContext(metadata={"initial": {"counter": {"count": 0}}}))

    assert context.metadata["answer"] == {"counter": {"count": 1}}
    assert "script_results" not in context.metadata
    assert_children_cleaned(docker_fixture, context, count=1)


@pytest.mark.asyncio
async def test_repeat_until_feedback_uses_fresh_containers_and_cleans_each_iteration(docker_fixture):
    repeated = RepeatUntilTaskStep(
        id="repeat", version=None, task=child_task(docker_fixture, script=COUNTER_SCRIPT),
        child_timeout_seconds=CHILD_TIMEOUT_SECONDS, cleanup_timeout_seconds=CLEANUP_TIMEOUT_SECONDS,
        max_iterations=2, until_key="script_results", until_value={"counter": {"count": 2}},
        input_bindings={"script_results": "initial"}, feedback_bindings={"script_results": "script_results"},
        output_bindings={"answer": "script_results"},
    )
    parent = Task(id="nested-repeat-parent", version=1, steps=[repeated])

    async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
        context = await parent.run(context=TaskStepContext(metadata={"initial": {"counter": {"count": 0}}}))

    assert context.metadata["answer"] == {"counter": {"count": 2}}
    invocations = get_child_invocation_store().list_for_parent(context.instance_id)
    sandbox_ids = [invocation.outcome["deployed_sandboxes"][0]["sandbox_id"] for invocation in invocations]
    assert len(set(sandbox_ids)) == 2
    assert_children_cleaned(docker_fixture, context, count=2)


@pytest.mark.asyncio
async def test_child_script_failure_removes_real_container_without_publishing_parent_outputs(docker_fixture):
    parent = Task(id="nested-failure-parent", version=1, steps=[subdag_step(child_task(docker_fixture, script=FAILURE_SCRIPT))])
    context = TaskStepContext(metadata={"initial": {"counter": {"count": 0}}})

    with pytest.raises(ChildTaskFailed):
        async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
            await parent.run(context=context)

    assert "answer" not in context.metadata
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.outcome is None
    assert get_task_instance_store().get(invocation.child_instance_id).status == "failed"
    assert_children_cleaned(docker_fixture, context, count=1)


async def await_running_script(fixture: DockerFixture) -> None:
    async with asyncio.timeout(READINESS_TIMEOUT_SECONDS):
        while True:
            containers = await asyncio.to_thread(fixture.container_ids)
            if containers:
                status = await asyncio.to_thread(
                    subprocess.run, ["docker", "exec", containers[0], "test", "-f", "/tmp/nested-child-ready"],
                    capture_output=True, timeout=DOCKER_COMMAND_TIMEOUT_SECONDS,
                )
                if status.returncode == 0:
                    return


@pytest.mark.asyncio
async def test_parent_cancellation_drains_blocking_native_child_and_removes_its_container(docker_fixture):
    parent = Task(id="nested-cancel-parent", version=1, steps=[subdag_step(child_task(docker_fixture, script=BLOCKING_SCRIPT))])
    context = TaskStepContext(metadata={"initial": {"counter": {"count": 0}}})
    running = asyncio.create_task(parent.run(context=context))
    try:
        await await_running_script(docker_fixture)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(CLEANUP_TIMEOUT_SECONDS + CHILD_TIMEOUT_SECONDS):
                await running
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert "answer" not in context.metadata
    invocation = get_child_invocation_store().list_for_parent(context.instance_id)[0]
    assert invocation.outcome is None
    assert get_task_instance_store().get(invocation.child_instance_id).status == "cancelled"
    assert_children_cleaned(docker_fixture, context, count=1)
