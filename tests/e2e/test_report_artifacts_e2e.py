"""Report artifact E2E regression tests."""

from __future__ import annotations

import sys
import types

import pytest


@pytest.mark.e2e
def test_compare_cli_records_compare_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest
    from model_optimizer.compare import cli as compare_cli

    output_dir = tmp_path / "compare"
    plot_output = output_dir / "compare.png"

    monkeypatch.setattr(compare_cli, "load_saved_data", lambda _path: [{"x": 1}])
    monkeypatch.setattr(
        compare_cli,
        "compare_predictions",
        lambda *_args, **_kwargs: {"x": {"l1_mean": 0.1, "l1_max": 0.2}},
    )

    def fake_plot(_metrics, path, **_kwargs):
        path = output_dir / "compare.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")

    monkeypatch.setattr(compare_cli, "plot_compare_results", fake_plot)

    compare_cli.compare_cli(
        [
            "workflow",
            "--data_path1",
            "/tmp/a.npz",
            "--data_path2",
            "/tmp/b.npz",
            "--plot_output",
            str(plot_output),
            "--output_dir",
            str(output_dir),
        ]
    )

    manifest = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert manifest.artifact_type == "compare_report"
    assert manifest.paths["data_path1"] == "/tmp/a.npz"
    assert manifest.paths["plot"] == str(plot_output)
    assert manifest.metrics["compare"]["x"]["l1_mean_avg"] == 0.1


@pytest.mark.e2e
def test_profile_cli_records_profile_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest
    from model_optimizer.profile import cli as profile_cli

    output_dir = tmp_path / "profile"

    def fake_profile(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        engine = output_dir / "llm.engine"
        e2e = output_dir / "llm_e2e.profile"
        layer = output_dir / "llm_layer.profile"
        engine.write_bytes(b"engine")
        e2e.write_text("e2e")
        layer.write_text("layer")
        return {
            "engine": str(engine),
            "e2e_profile": str(e2e),
            "layer_profile": str(layer),
        }

    monkeypatch.setattr(profile_cli, "profile_onnx", fake_profile)

    profile_cli.profile_cli(
        [
            "workflow",
            "--model_path",
            "/models/llm.onnx",
            "--output_dir",
            str(output_dir),
            "--e2e_profile",
            "True",
            "--layer_profile",
            "True",
        ]
    )

    manifest = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert manifest.artifact_type == "profile_report"
    assert manifest.paths["engine"].endswith("llm.engine")
    assert manifest.paths["e2e_profile"].endswith("llm_e2e.profile")
    assert manifest.paths["layer_profile"].endswith("llm_layer.profile")
    assert manifest.configs["profile"]["e2e_profile"] is True


@pytest.mark.e2e
def test_eval_cli_uses_runner_and_records_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest
    from model_optimizer.evaluate.cli import eval_cli

    output_dir = tmp_path / "eval"

    class FakeMetric:
        def get_result(self):
            return {"ignored": "large result"}

    class FakeModel:
        @classmethod
        def construct_from_name_path(cls, model_name, model_path, train_config=None):
            _ = (model_name, model_path, train_config)
            return cls()

        def val(self, dataset, batch_size, max_data, output_dir):
            _ = (dataset, batch_size, max_data)
            output_dir = output_dir
            return FakeMetric()

    fake_registry = types.ModuleType("model_optimizer.models.registry")
    fake_registry.get_model_cls = lambda _name: FakeModel
    monkeypatch.setitem(sys.modules, "model_optimizer.models.registry", fake_registry)

    eval_cli(
        [
            "workflow",
            "--model_name",
            "pi05_libero/llm",
            "--model_path",
            "/models/pi05",
            "--dataset",
            "/data/eval",
            "--output_dir",
            str(output_dir),
            "--batch_size",
            "2",
            "--max_data",
            "3",
        ]
    )

    manifest = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert manifest.artifact_type == "eval_report"
    assert manifest.architecture == "pi05"
    assert manifest.stage == "llm"
    assert manifest.configs["eval"]["batch_size"] == 2
    assert manifest.metrics["eval"] == {
        "has_metric": True,
        "metric_type": "FakeMetric",
    }


@pytest.mark.e2e
def test_mock_second_architecture_minimal_registry_chain(tmp_path):
    from model_optimizer.architectures import ArchitectureSpec, StageSpec, register_architecture_spec
    from model_optimizer.calibrate.collectors import CalibCollector, get_calib_collector, register_calib_collector
    from model_optimizer.evaluate.metrics import Metric, create_metric, register_metric
    from model_optimizer.workflows import WorkflowManifest, WorkflowRunner

    register_architecture_spec(
        ArchitectureSpec(
            name="world_model_acceptance_e2e",
            stages=(StageSpec(name="encoder", supported_backends=("pytorch", "tensorrt")),),
        )
    )

    class EncoderCollector(CalibCollector):
        def load(self, calib_data: str):
            return [{"path": calib_data}]

    class EncoderMetric(Metric):
        def print(self):
            return None

        def compare(self, other):
            return (self.get_result(), other.get_result())

    register_calib_collector(
        EncoderCollector(architecture="world_model_acceptance_e2e", stage="encoder")
    )
    register_metric("world_model_acceptance_e2e", "encoder", EncoderMetric)

    assert get_calib_collector("world_model_acceptance_e2e", "encoder").load("/calib") == [
        {"path": "/calib"}
    ]
    assert isinstance(
        create_metric("world_model_acceptance_e2e", "encoder", {"ok": True}),
        EncoderMetric,
    )

    manifest = WorkflowManifest.from_dict(
        {
            "version": "workflow_v1",
            "architecture": "world_model_acceptance_e2e",
            "model_name": "world_model",
            "model_path": "/models/world",
            "output_dir": str(tmp_path / "workflow"),
            "stages": [
                {
                    "name": "encoder",
                    "actions": ["export"],
                }
            ],
        }
    )
    plan = WorkflowRunner(manifest).plan()
    assert plan.commands[0].stage == "encoder"
    assert plan.commands[0].action == "export"
    assert "world_model/encoder" in plan.commands[0].argv
