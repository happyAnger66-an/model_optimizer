"""Workflow manifest E2E regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_workflow(path: Path, output_dir: Path, *, dry_run: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "workflow_v1",
                "architecture": "pi05",
                "model_name": "pi05_libero",
                "model_path": "/models/pi05",
                "output_dir": str(output_dir),
                "dry_run": dry_run,
                "stages": [
                    {
                        "name": "llm",
                        "actions": ["quantize", "build"],
                        "quantize": {
                            "quantize_cfg": "config/quant/llm_quant_fp8_cfg.py",
                            "calibrate_data": "/tmp/calib.pt",
                        },
                        "build": {
                            "build_cfg": "config/build_configs/llm_build_cfg.py",
                        },
                    },
                    {
                        "name": "expert",
                        "actions": ["export"],
                        "export": {"mode": "native_per_layer"},
                    },
                ],
            }
        )
    )


@pytest.mark.e2e
def test_workflow_manifest_plan_builds_expected_commands(tmp_path):
    from model_optimizer.workflows import WorkflowManifest, WorkflowRunner

    manifest_path = tmp_path / "workflow.json"
    _write_workflow(manifest_path, tmp_path / "artifacts")

    manifest = WorkflowManifest.load(manifest_path)
    result = WorkflowRunner(manifest).plan()

    assert [(c.stage, c.action) for c in result.commands] == [
        ("llm", "quantize"),
        ("llm", "build"),
        ("expert", "export"),
    ]
    assert result.commands[0].argv[:5] == [
        "workflow",
        "--model_name",
        "pi05_libero/llm",
        "--model_path",
        "/models/pi05",
    ]
    assert "--quantize_cfg" in result.commands[0].argv
    assert result.commands[1].argv[result.commands[1].argv.index("--model_path") + 1].endswith(
        "artifacts/llm/llm.onnx"
    )
    assert result.commands[2].argv[result.commands[2].argv.index("--model_name") + 1] == "pi05_libero/expert"


@pytest.mark.e2e
def test_workflow_runner_dry_run_does_not_execute_steps(tmp_path, monkeypatch):
    from model_optimizer.workflows import WorkflowManifest, WorkflowRunner

    manifest_path = tmp_path / "workflow.json"
    _write_workflow(manifest_path, tmp_path / "artifacts", dry_run=True)
    called = []

    def fail_if_called(_cmd):
        called.append(_cmd)
        raise AssertionError("dry_run should not execute step functions")

    monkeypatch.setattr(WorkflowRunner, "_run_command", fail_if_called)

    result = WorkflowRunner(WorkflowManifest.load(manifest_path)).run()

    assert len(result.commands) == 3
    assert called == []


@pytest.mark.e2e
def test_workflow_runner_executes_fake_steps_and_records_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest, manifest_path_for
    from model_optimizer.workflows import WorkflowManifest, WorkflowRunner

    manifest_path = tmp_path / "workflow.json"
    _write_workflow(manifest_path, tmp_path / "artifacts", dry_run=False)
    calls = []

    def fake_run_command(_runner, cmd):
        calls.append((cmd.action, cmd.argv))
        if cmd.action in ("quantize", "export"):
            out = Path(cmd.argv[cmd.argv.index("--export_dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{cmd.stage}.onnx").write_bytes(b"onnx")
            ArtifactManifest(
                artifact_type="quantized_onnx" if cmd.action == "quantize" else "onnx_export",
                paths={"onnx": str((out / f"{cmd.stage}.onnx").resolve())},
            ).save(out / "artifact_manifest.json")
        if cmd.action == "build":
            engine = Path(cmd.argv[cmd.argv.index("--export_dir") + 1])
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(b"engine")

    monkeypatch.setattr(WorkflowRunner, "_run_command", fake_run_command)

    WorkflowRunner(WorkflowManifest.load(manifest_path)).run()

    assert [name for name, _ in calls] == ["quantize", "build", "export"]
    llm_manifest = ArtifactManifest.load(manifest_path_for(tmp_path / "artifacts" / "llm"))
    expert_manifest = ArtifactManifest.load(manifest_path_for(tmp_path / "artifacts" / "expert"))
    assert llm_manifest.stage == "llm"
    assert llm_manifest.artifact_type == "quantized_onnx"
    assert llm_manifest.paths["onnx"].endswith("llm.onnx")
    assert [a["name"] for a in llm_manifest.metadata["workflow"]["actions"]] == ["quantize", "build"]
    assert expert_manifest.stage == "expert"
    assert expert_manifest.artifact_type == "onnx_export"
    assert expert_manifest.paths["onnx"].endswith("expert.onnx")
    assert [a["name"] for a in expert_manifest.metadata["workflow"]["actions"]] == ["export"]


@pytest.mark.e2e
def test_workflow_cli_plan(cli_cmd, tmp_path):
    manifest_path = tmp_path / "workflow.json"
    _write_workflow(manifest_path, tmp_path / "artifacts")

    result = cli_cmd("workflow", "plan", "--manifest", str(manifest_path))

    assert result.returncode == 0
    assert "[workflow] llm.quantize:" in result.stdout
    assert "[workflow] llm.build:" in result.stdout
    assert "[workflow] expert.export:" in result.stdout
