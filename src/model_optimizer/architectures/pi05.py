"""Pi0.5 architecture specification."""

from __future__ import annotations

from .base import ArchitectureSpec, StageSpec
from .registry import register_architecture_spec

PI05_ARCHITECTURE_NAME = "pi05"

PI05_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        name="vit",
        supported_backends=("pytorch", "tensorrt", "onnxrt", "flashrt"),
        quantizable=True,
        description="SigLIP vision encoder stage.",
    ),
    StageSpec(
        name="embed_prefix",
        supported_backends=("pytorch", "tensorrt", "onnxrt"),
        quantizable=True,
        description="Image and language prefix embedding stage.",
    ),
    StageSpec(
        name="llm",
        supported_backends=("pytorch", "tensorrt", "onnxrt"),
        quantizable=True,
        description="PaliGemma prefix language-model stage.",
    ),
    StageSpec(
        name="expert",
        supported_backends=("pytorch", "tensorrt", "onnxrt", "native"),
        quantizable=True,
        description="Gemma action expert stage.",
    ),
    StageSpec(
        name="denoise",
        supported_backends=("pytorch", "tensorrt", "onnxrt", "native", "flashrt"),
        quantizable=True,
        description="Flow-matching denoise loop or denoise step.",
    ),
)

PI05_ARCHITECTURE_SPEC = ArchitectureSpec(
    name=PI05_ARCHITECTURE_NAME,
    stages=PI05_STAGE_SPECS,
    description="OpenPI Pi0.5 policy architecture.",
)


def register() -> ArchitectureSpec:
    """Register and return the Pi0.5 architecture spec."""

    return register_architecture_spec(PI05_ARCHITECTURE_SPEC)


register()
