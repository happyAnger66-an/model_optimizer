"""Artifact manifest E2E regression tests."""

from __future__ import annotations

import json

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
