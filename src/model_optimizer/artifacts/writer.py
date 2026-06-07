"""Helpers for writing artifact manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import ArtifactManifest, MANIFEST_FILENAME


def infer_stage_from_model_name(model_name: str | None) -> str:
    if not model_name:
        return ""
    return str(model_name).split("/")[-1] if "/" in str(model_name) else ""


def infer_architecture_from_model_name(model_name: str | None) -> str:
    if not model_name:
        return ""
    root = str(model_name).split("/")[0]
    if root.startswith("pi05"):
        return "pi05"
    return root


def find_onnx_artifact(output_dir: str | Path, *, stage: str | None = None) -> Path | None:
    """Find the ONNX artifact in an output directory.

    Prefer ``<stage>.onnx`` when stage is known, otherwise return a single ONNX
    file only when the directory is unambiguous.
    """

    directory = Path(output_dir).expanduser()
    if not directory.is_dir():
        return None
    if stage:
        preferred = directory / f"{stage}.onnx"
        if preferred.is_file():
            return preferred
    matches = sorted(directory.glob("*.onnx"))
    if len(matches) == 1:
        return matches[0]
    return None


def manifest_path_for(output_path: str | Path) -> Path:
    """Return the manifest path for an artifact file or output directory."""

    p = Path(output_path).expanduser()
    directory = p if p.suffix == "" else p.parent
    return directory / MANIFEST_FILENAME


def load_or_create_manifest(path: str | Path) -> ArtifactManifest:
    p = Path(path).expanduser()
    if p.is_file():
        return ArtifactManifest.load(p)
    return ArtifactManifest()


def update_artifact_manifest(
    output_path: str | Path,
    *,
    artifact_type: str | None = None,
    architecture: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    precision: str | None = None,
    quant_format: str | None = None,
    paths: dict[str, str] | None = None,
    configs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Create or update ``artifact_manifest.json`` next to ``output_path``."""

    manifest_path = manifest_path_for(output_path)
    manifest = load_or_create_manifest(manifest_path)
    manifest.merge_update(
        artifact_type=artifact_type,
        architecture=architecture,
        stage=stage,
        model_name=model_name,
        precision=precision,
        quant_format=quant_format,
        paths=paths,
        configs=configs,
        metrics=metrics,
        metadata=metadata,
    )
    manifest.save(manifest_path)
    return manifest


def record_trt_build_artifact(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str,
    build_time_s: float,
    engine_size_mb: float,
    use_cudagraph: bool,
    build_config: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Record a TensorRT engine build result."""

    onnx_p = Path(onnx_path).expanduser().resolve()
    engine_p = Path(engine_path).expanduser().resolve()
    return update_artifact_manifest(
        engine_p,
        artifact_type="tensorrt_engine",
        precision=str(precision),
        paths={
            "onnx": str(onnx_p),
            "engine": str(engine_p),
        },
        configs={
            "build": dict(build_config or {}),
        },
        metrics={
            "build_time_s": float(build_time_s),
            "engine_size_mb": float(engine_size_mb),
        },
        metadata={
            "use_cudagraph": bool(use_cudagraph),
        },
    )


def record_onnx_export_artifact(
    *,
    output_dir: str | Path,
    onnx_path: str | Path | None,
    architecture: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    export_config: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Record an ONNX export result."""

    resolved_stage = stage or infer_stage_from_model_name(model_name)
    resolved_onnx = Path(onnx_path).expanduser() if onnx_path else find_onnx_artifact(output_dir, stage=resolved_stage)
    paths: dict[str, str] = {}
    if model_path:
        paths["model"] = str(Path(model_path).expanduser())
    if resolved_onnx is not None:
        paths["onnx"] = str(resolved_onnx.resolve())

    return update_artifact_manifest(
        output_dir,
        artifact_type="onnx_export",
        architecture=architecture,
        stage=resolved_stage,
        model_name=model_name,
        paths=paths,
        configs={
            "export": dict(export_config or {}),
        },
    )


def record_quantize_artifact(
    *,
    output_dir: str | Path,
    architecture: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    quantize_config: dict[str, Any] | None = None,
    onnx_path: str | Path | None = None,
) -> ArtifactManifest:
    """Record a quantization result."""

    resolved_stage = stage or infer_stage_from_model_name(model_name)
    resolved_onnx = Path(onnx_path).expanduser() if onnx_path else find_onnx_artifact(output_dir, stage=resolved_stage)
    paths: dict[str, str] = {}
    if model_path:
        paths["model"] = str(Path(model_path).expanduser())
    if resolved_onnx is not None:
        paths["onnx"] = str(resolved_onnx.resolve())

    return update_artifact_manifest(
        output_dir,
        artifact_type="quantized_onnx",
        architecture=architecture,
        stage=resolved_stage,
        model_name=model_name,
        paths=paths,
        configs={
            "quantize": dict(quantize_config or {}),
        },
    )
