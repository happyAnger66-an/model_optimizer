"""Artifact manifest E2E regression tests."""

from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.mark.e2e
def test_update_artifact_manifest_merges_existing_data(tmp_path):
    from model_optimizer.artifacts import ArtifactManifest, manifest_path_for, update_artifact_manifest

    engine_path = tmp_path / "build" / "llm.engine"
    manifest_path = manifest_path_for(engine_path)
    ArtifactManifest(
        artifact_type="onnx_export",
        paths={"onnx": "/tmp/old.onnx"},
        metadata={"source": "export"},
    ).save(manifest_path)

    manifest = update_artifact_manifest(
        engine_path,
        artifact_type="tensorrt_engine",
        precision="bf16",
        paths={"engine": str(engine_path)},
        metrics={"engine_size_mb": 12.5},
        metadata={"builder": "trt"},
    )

    loaded = ArtifactManifest.load(manifest_path)
    assert manifest.to_dict() == loaded.to_dict()
    assert loaded.artifact_type == "tensorrt_engine"
    assert loaded.precision == "bf16"
    assert loaded.paths["onnx"] == "/tmp/old.onnx"
    assert loaded.paths["engine"] == str(engine_path)
    assert loaded.metrics["engine_size_mb"] == 12.5
    assert loaded.metadata == {"source": "export", "builder": "trt"}


@pytest.mark.e2e
def test_record_trt_build_artifact_writes_manifest(tmp_path):
    from model_optimizer.artifacts import MANIFEST_FILENAME, record_trt_build_artifact

    onnx_path = tmp_path / "export" / "llm.onnx"
    engine_path = tmp_path / "build" / "llm.engine"
    onnx_path.parent.mkdir()
    engine_path.parent.mkdir()
    onnx_path.write_bytes(b"onnx")
    engine_path.write_bytes(b"engine")

    manifest = record_trt_build_artifact(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision="fp16",
        build_time_s=1.25,
        engine_size_mb=0.01,
        use_cudagraph=True,
        build_config={
            "workspace_mb": 1024,
            "min_shapes": {"input": (1, 4)},
        },
    )

    manifest_path = engine_path.parent / MANIFEST_FILENAME
    raw = json.loads(manifest_path.read_text())
    assert raw == manifest.to_dict()
    assert raw["artifact_type"] == "tensorrt_engine"
    assert raw["precision"] == "fp16"
    assert raw["paths"]["onnx"] == str(onnx_path.resolve())
    assert raw["paths"]["engine"] == str(engine_path.resolve())
    assert raw["configs"]["build"]["workspace_mb"] == 1024
    assert raw["configs"]["build"]["min_shapes"]["input"] == [1, 4]
    assert raw["metrics"]["build_time_s"] == 1.25
    assert raw["metadata"]["use_cudagraph"] is True


@pytest.mark.e2e
def test_record_quantize_artifact_finds_stage_onnx(tmp_path):
    from model_optimizer.artifacts import ArtifactManifest, infer_architecture_from_model_name, record_quantize_artifact

    output_dir = tmp_path / "quant"
    output_dir.mkdir()
    onnx_path = output_dir / "llm.onnx"
    onnx_path.write_bytes(b"onnx")

    record_quantize_artifact(
        output_dir=output_dir,
        architecture=infer_architecture_from_model_name("pi05_libero/llm"),
        model_name="pi05_libero/llm",
        model_path="/models/pi05",
        quantize_config={
            "quantize_cfg": "config/quant/llm_quant_fp8_cfg.py",
            "calibrate_data": "/data/calib",
        },
    )

    loaded = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert loaded.artifact_type == "quantized_onnx"
    assert loaded.architecture == "pi05"
    assert loaded.stage == "llm"
    assert loaded.model_name == "pi05_libero/llm"
    assert loaded.paths["onnx"] == str(onnx_path.resolve())
    assert loaded.configs["quantize"]["calibrate_data"] == "/data/calib"


@pytest.mark.e2e
def test_convert_cli_records_export_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest
    from model_optimizer.convert.convert_formt import convert_model

    output_dir = tmp_path / "export"

    class FakeModel:
        applied_features = ["fused_mlp"]

        @classmethod
        def construct_from_name_path(cls, model_name, model_path, train_config=None, feature_config=None):
            _ = (model_name, model_path, train_config, feature_config)
            return cls()

        def export(self, export_dir, mode="native_per_layer"):
            _ = mode
            out = tmp_path / "export" / "llm.onnx"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx")
            return str(out)

    fake_registry = types.ModuleType("model_optimizer.models.registry")
    fake_registry.get_model_cls = lambda _name: FakeModel
    monkeypatch.setitem(sys.modules, "model_optimizer.models.registry", fake_registry)

    convert_model(
        [
            "workflow",
            "--model_name",
            "pi05_libero/llm",
            "--model_path",
            "/models/pi05",
            "--export_dir",
            str(output_dir),
            "--mode",
            "native_per_layer",
        ]
    )

    loaded = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert loaded.artifact_type == "onnx_export"
    assert loaded.architecture == "pi05"
    assert loaded.stage == "llm"
    assert loaded.paths["onnx"].endswith("llm.onnx")
    assert loaded.configs["export"]["mode"] == "native_per_layer"
    assert loaded.configs["export"]["applied_features"] == ["fused_mlp"]


@pytest.mark.e2e
def test_quantize_cli_records_quantize_manifest(tmp_path, monkeypatch):
    from model_optimizer.artifacts import ArtifactManifest
    from model_optimizer.quantization import cli as quant_cli

    output_dir = tmp_path / "quant"

    class FakeModel:
        applied_features = ["fused_mlp"]

        @classmethod
        def construct_from_name_path(cls, model_name, model_path, train_config=None, feature_config=None):
            _ = (model_name, model_path, train_config, feature_config)
            return cls()

        def quantize(self, quant_cfg, calibrate_data, export_dir, *, measure_quant_error=False):
            _ = (quant_cfg, calibrate_data, measure_quant_error)
            out = tmp_path / "quant" / "llm.onnx"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx")

    fake_registry = types.ModuleType("model_optimizer.models.registry")
    fake_registry.get_model_cls = lambda _name: FakeModel
    monkeypatch.setitem(sys.modules, "model_optimizer.models.registry", fake_registry)
    monkeypatch.setattr(quant_cli, "get_quant_cfg", lambda _path: {"format": "fp8"})

    quant_cli.quantize_cli(
        [
            "workflow",
            "--model_name",
            "pi05_libero/llm",
            "--model_path",
            "/models/pi05",
            "--quantize_cfg",
            "config/quant/llm_quant_fp8_cfg.py",
            "--calibrate_data",
            "/data/calib",
            "--export_dir",
            str(output_dir),
        ]
    )

    loaded = ArtifactManifest.load(output_dir / "artifact_manifest.json")
    assert loaded.artifact_type == "quantized_onnx"
    assert loaded.architecture == "pi05"
    assert loaded.stage == "llm"
    assert loaded.paths["onnx"].endswith("llm.onnx")
    assert loaded.configs["quantize"]["quantize_cfg"] == "config/quant/llm_quant_fp8_cfg.py"
    assert loaded.configs["quantize"]["applied_features"] == ["fused_mlp"]
