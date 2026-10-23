"""Embedded task entities resolve, plan and export within their own output scopes."""

import json

import pytest

from agent_env.artifact import FileArtifact
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.importer import import_task
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.preflight import preflight_run
from agent_env.bundle.resolve import build_step, resolve_bundle
from agent_env.config import configure
from agent_env.entity.refs import EntityKind
from agent_env.entity.scopes import FieldScope
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.traversal import walk_step_definitions
from tst.unit.bundle._support import layout


def nested_step(step_id, children, *, step_type="subdag"):
    value = {
        "id": step_id, "type": step_type, "child_timeout_seconds": 10, "cleanup_timeout_seconds": 5,
        "task": {"id": "embedded-child", "version": None, "steps": children},
    }
    if step_type == "repeat_until":
        value.update(max_iterations=3, until_key="done", until_value=True)
    return value


def bundle_root(tmp_path, steps, files=None):
    return layout(tmp_path / "nested", {
        "tasks/root.json": json.dumps(steps), **(files or {}),
    })


def root_task(resolved):
    return next(entry for entry in resolved.entries if entry.entry.name == "root")


def read_bundle(folder):
    return parse_bundle(folder, id_root="@local/tests/nested")


def test_nested_bundle_rewrites_references_at_each_depth_without_external_child_task_reference(tmp_path):
    steps = [nested_step("outer", [nested_step("loop", [
        {"id": "load", "type": "load_artifact", "artifact_id": "fixture", "env_id": "db"},
    ], step_type="repeat_until")])]
    folder = bundle_root(tmp_path, steps, {
        "artifacts/fixture/data.json": "{}", "envs/db/env.toml": 'environment_name = "db"\n',
        "envs/db/Dockerfile": "FROM scratch\n",
    })

    task = root_task(resolve_bundle(read_bundle(folder)))

    child = task.config[0]["task"]["steps"][0]["task"]["steps"][0]
    assert child["artifact_id"] == "@local/tests/nested/fixture"
    assert child["env_id"] == "@local/tests/nested/db"
    assert {ref.kind for ref in task.references} == {EntityKind.ARTIFACT, EntityKind.ENV}
    assert all(ref.id != "embedded-child" for ref in task.references)
    assert all("step 'outer': task: step 'loop': task:" in ref.where for ref in task.references)


def test_sibling_child_graphs_keep_same_named_outputs_and_dependencies_isolated(tmp_path):
    children = [
        {"id": "build", "type": "build_mcp_cli", "env_id": "db", "command_name": "db", "cli_artifact_id": "client"},
        {"id": "read", "type": "add_skills", "cli_artifact_ids": ["client"], "depends_on": ["build"]},
    ]
    folder = bundle_root(tmp_path, [nested_step("first", children), nested_step("second", children)], {
        "envs/db/env.toml": 'environment_name = "db"\n', "envs/db/Dockerfile": "FROM scratch\n",
    })

    task = root_task(resolve_bundle(read_bundle(folder)))

    outputs = [step["task"]["steps"][0]["cli_artifact_id"] for step in task.config]
    assert outputs == ["@local/tests/nested/root/first/task/client", "@local/tests/nested/root/second/task/client"]
    assert [step["task"]["steps"][1]["cli_artifact_ids"] for step in task.config] == [[value] for value in outputs]
    assert {output.id for output in task.outputs} == set(outputs)
    assert all(ref.kind is EntityKind.ENV for ref in task.references)


def test_nested_output_dependency_violation_is_not_hidden_by_outer_sequential_order(tmp_path):
    children = [
        {"id": "build", "type": "build_mcp_cli", "env_id": "db", "command_name": "db", "cli_artifact_id": "client"},
        {"id": "read", "type": "add_skills", "cli_artifact_ids": ["client"], "depends_on": []},
    ]
    folder = bundle_root(tmp_path, [nested_step("outer", children)], {
        "envs/db/env.toml": 'environment_name = "db"\n', "envs/db/Dockerfile": "FROM scratch\n",
    })

    with pytest.raises(BundleError, match="doesn't depend on"):
        resolve_bundle(read_bundle(folder))


def test_nested_reference_enters_plan_closure_and_child_task_id_does_not(tmp_path):
    folder = bundle_root(tmp_path, [nested_step("outer", [
        {"id": "load", "type": "load_artifact", "artifact_id": "fixture", "sandbox_name": "runtime"},
    ])], {"artifacts/fixture/data.json": "{}"})

    with namespace_routing():
        plan = plan_bundle(resolve_bundle(read_bundle(folder)))

    assert {write.id for write in plan.writes} == {"@local/tests/nested/fixture", "@local/tests/nested/root"}
    root = next(write for write in plan.writes if write.id.endswith("/root"))
    assert root.needs == (("artifact", "@local/tests/nested/fixture"),)


def test_nested_task_export_writes_referenced_artifact_and_prunes_child_user_scoped_fields(local_stores, tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}")
    FileArtifact.put(id="fixture", description="test fixture", file_path=str(fixture))
    child = nested_step("outer", [
        {"id": "load", "type": "load_artifact", "artifact_id": "fixture", "sandbox_name": "runtime",
         "urls": ["https://fixture.invalid/private.json"]},
    ])
    Task.put(id="root", steps=[build_step(child)])

    imported = import_task("root", tmp_path / "export", include_scopes=[FieldScope.TASK])

    exported = json.loads((imported.root / "tasks/root.json").read_text())
    assert {written.id for written in imported.written} == {"fixture"}
    assert "urls" not in exported[0]["task"]["steps"][0]
    assert len(parse_bundle(imported.root).entries) == 2


def test_embedded_task_traversal_uses_declaration_for_custom_step_types():
    class CustomNestedStep(TaskStep):
        type = "custom_nested"
        embedded_task_fields = ("child_graph",)

    registry = {**get_task_step_registry(), CustomNestedStep.type: CustomNestedStep}
    steps = [{"id": "outer", "type": CustomNestedStep.type,
              "child_graph": {"id": "child", "steps": [{"id": "leaf", "type": "review"}]}}]

    visited = list(walk_step_definitions(steps, registry))

    assert [(location.path, location.value["id"]) for location in visited] == [
        ("steps[0]", "outer"), ("steps[0].child_graph.steps[0]", "leaf"),
    ]


def test_nested_default_agent_reference_is_discovered_before_plan_writes(tmp_path):
    configure(default_a2a_agent_id="@local/tests/nested/solver")
    folder = bundle_root(tmp_path, [nested_step("outer", [{"id": "deploy", "type": "deploy_agent"}])], {
        "agents/solver/Dockerfile": "FROM scratch\n",
    })

    with namespace_routing():
        plan = plan_bundle(resolve_bundle(read_bundle(folder)))

    assert "@local/tests/nested/solver" in {write.id for write in plan.writes}


def test_nested_deployment_provider_requirements_are_checked_before_any_write(tmp_path, monkeypatch):
    folder = bundle_root(tmp_path, [nested_step("outer", [
        {"id": "sandbox", "type": "deploy_sandbox", "sandbox_name": "runtime",
         "sandbox_mode": "container", "image": "python:3.12-slim"},
    ])])
    monkeypatch.setattr("agent_env.bundle.preflight.docker_unreachable", lambda: "Docker is unavailable")

    with namespace_routing():
        plan = plan_bundle(resolve_bundle(read_bundle(folder)))
        with pytest.raises(BundleError, match="Docker is unavailable") as failure:
            preflight_run(plan, plan.tasks, sandbox="local")

    assert "step 'outer': task: step 'sandbox'" in str(failure.value)


def test_nested_cli_install_checks_its_own_agent_and_environment_requirements(tmp_path):
    folder = bundle_root(tmp_path, [nested_step("outer", [
        {"id": "load", "type": "load_artifact", "artifact_id": "client", "sandbox_name": "runtime"},
    ])], {
        "artifacts/client/artifact.toml": 'type = "cli"\ncommand_name = "client"\nentrypoint = "client.py"\n',
        "artifacts/client/client.py": "print('client')\n",
    })

    with namespace_routing():
        with pytest.raises(BundleError, match="needs agent_name and env_id") as failure:
            plan_bundle(resolve_bundle(read_bundle(folder)))

    assert "step 'outer': task: step 'load'" in str(failure.value)


def test_same_cli_command_in_separate_child_contexts_does_not_collide(tmp_path):
    children = [{"id": "load", "type": "load_artifact", "artifact_id": "client",
                 "agent_name": "solver", "env_id": "db"}]
    folder = bundle_root(tmp_path, [nested_step("first", children), nested_step("second", children)], {
        "artifacts/client/artifact.toml": 'type = "cli"\ncommand_name = "client"\nentrypoint = "client.py"\n',
        "artifacts/client/client.py": "print('client')\n",
        "envs/db/env.toml": 'environment_name = "db"\n', "envs/db/Dockerfile": "FROM scratch\n",
    })

    with namespace_routing():
        plan = plan_bundle(resolve_bundle(read_bundle(folder)))

    assert len(plan.tasks) == 1


def test_nested_loaded_environment_names_are_checked_against_the_childs_target_env(tmp_path):
    folder = bundle_root(tmp_path, [nested_step("outer", [
        {"id": "load", "type": "load_artifact", "artifact_id": "state", "env_id": "db"},
    ])], {
        "artifacts/state/artifact.toml": 'type = "environment"\nenvironment_name = "wrong-server"\n',
        "artifacts/state/data.json": "{}",
        "envs/db/env.toml": 'environment_name = "db"\n', "envs/db/Dockerfile": "FROM scratch\n",
    })

    with namespace_routing():
        with pytest.raises(BundleError, match="wrong-server") as failure:
            plan_bundle(resolve_bundle(read_bundle(folder)))

    assert "step 'outer': task: step 'load'" in str(failure.value)


def test_dry_run_defers_nested_preflight_until_the_referenced_artifact_is_written(tmp_path):
    folder = bundle_root(tmp_path, [nested_step("outer", [
        {"id": "load", "type": "load_artifact", "artifact_id": "fixture", "sandbox_name": "runtime"},
    ])], {"artifacts/fixture/data.json": "{}"})

    with namespace_routing():
        plan = plan_bundle(resolve_bundle(read_bundle(folder)))
        result = materialize(plan, dry_run=True)

    assert [(write.id, step.id) for write, step in result.not_preflighted] == [("@local/tests/nested/root", "outer")]
