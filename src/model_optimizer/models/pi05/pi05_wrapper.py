"""Pi0.5 whole-graph FP8/NVFP4 quantization and ONNX export.

This implementation is self-contained under `model_optimizer.models.pi05.utils`
and does NOT depend on `openpi_on_thor` or `third_party/openpi_on_thor` at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import torch

from model_optimizer.models.model import Model
from model_optimizer.models.pi05.utils.whole_export import (
    ONNXWrapper,
    create_dummy_inputs,
    export_whole_model_to_onnx,
    prepare_model_for_export,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


class Pi05Wrapper(torch.nn.Module, Model):
    """Wraps ``PI0Pytorch`` for end-to-end FP8 quantization and single ONNX export."""

    def __init__(
        self,
        pi05_model: torch.nn.Module,
        *,
        config_obj: Any = None,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "pi05_droid",
        model_path: str = "",
    ):
        torch.nn.Module.__init__(self)
        Model.__init__(self, model_name, model_path or (checkpoint_dir or ""))
        self.pi05_model = pi05_model
        self._config_obj = config_obj
        self._checkpoint_dir = checkpoint_dir

    @property
    def model(self) -> torch.nn.Module:
        return self.pi05_model

    @classmethod
    def construct_from_name_path(cls, model_name: str, model_path: str) -> Pi05Wrapper:
        """Used by ``model-opt export --model_name pi05_libero/wrapper --model_path ...``.

        ``model_name`` must be ``<config_prefix>/wrapper`` (e.g. ``pi05_libero/wrapper``);
        ``model_path`` is the OpenPI PyTorch checkpoint directory.
        """
        from openpi.policies import policy_config
        from openpi.training import config as _config

        real_name = model_name.split("/")[0]
        print(f"pi05 wrapper: loading config {real_name} from {model_path}")
        config = _config.get_config(real_name)
        policy = policy_config.create_trained_policy(config, model_path)
        return cls(
            policy._model,
            config_obj=config,
            checkpoint_dir=model_path,
            model_name=real_name,
            model_path=model_path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        config_name: str = "pi05_droid",
    ) -> Pi05Wrapper:
        return cls.construct_from_name_path(f"{config_name}/wrapper", checkpoint_dir)

    def forward(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Same I/O contract as Thor ``ONNXWrapper`` (flat tensors → ``sample_actions``)."""
        return ONNXWrapper(self.pi05_model, num_steps)(images, img_masks, lang_tokens, lang_masks, state, noise)

    def quantize(
        self,
        *,
        num_steps: int = 10,
        precision: str = "fp8",
        config_obj: Any = None,
        checkpoint_dir: Optional[str] = None,
        num_calibration_samples: int = 32,
        enable_llm_nvfp4: bool = True,
        quantize_attention_matmul: bool = True,
    ) -> Pi05Wrapper:
        """FP8 (+ optional NVFP4 LLM) quantization via ModelOpt; matches ``pytorch_to_onnx``."""
        cfg = config_obj if config_obj is not None else self._config_obj
        ckpt = checkpoint_dir if checkpoint_dir is not None else self._checkpoint_dir

        device = next(self.pi05_model.parameters()).device
        dummy_inputs = create_dummy_inputs(device, self.pi05_model.config, torch.bfloat16)

        self.pi05_model = prepare_model_for_export(
            self.pi05_model,
            precision=precision,
            dummy_inputs=dummy_inputs,
            config_obj=cfg,
            checkpoint_dir=ckpt,
            num_calibration_samples=num_calibration_samples,
            num_steps=num_steps,
            enable_llm_nvfp4=enable_llm_nvfp4,
            quantize_attention_matmul=quantize_attention_matmul,
        )
        self.is_quantized = True
        return self

    def export(
        self,
        export_dir: Union[str, Path],
        mode: str = "native_per_layer",
        *,
        num_steps: Optional[int] = None,
        precision: Optional[str] = None,
        config_obj: Any = None,
        checkpoint_dir: Optional[str] = None,
        num_calibration_samples: Optional[int] = None,
        enable_llm_nvfp4: Optional[bool] = None,
        quantize_attention_matmul: Optional[bool] = None,
        quantize: Optional[bool] = None,
    ) -> str:
        """Export a single end-to-end ONNX under ``export_dir/onnx/`` (quantizes if needed).

        Matches ``model-opt export`` / ``convert_model``: first positional argument is
        ``export_dir``; ``mode`` is accepted for CLI compatibility (sub-model exporters use
        it; whole-graph export ignores it).

        Optional tuning via environment (defaults match Thor ``pytorch_to_onnx``):
        ``PI05_WHOLE_NUM_STEPS``, ``PI05_WHOLE_PRECISION``, ``PI05_WHOLE_NUM_CALIB_SAMPLES``,
        ``PI05_WHOLE_ENABLE_LLM_NVFP4``, ``PI05_WHOLE_QUANTIZE_ATTN_MATMUL`` (0/1),
        ``PI05_WHOLE_EXPORT_NO_QUANT`` (1 = ONNX export only, skip ModelOpt FP8 quantization).
        """
        _ = mode  # same signature as pi05 vit/llm/expert; not used for single ONNX graph

        num_steps = num_steps if num_steps is not None else _env_int("PI05_WHOLE_NUM_STEPS", 10)
        precision = precision if precision is not None else (os.environ.get("PI05_WHOLE_PRECISION") or "fp8")
        num_calibration_samples = num_calibration_samples if num_calibration_samples is not None else _env_int(
            "PI05_WHOLE_NUM_CALIB_SAMPLES", 32
        )
        if enable_llm_nvfp4 is None:
            enable_llm_nvfp4 = _env_bool("PI05_WHOLE_ENABLE_LLM_NVFP4", True)
        if quantize_attention_matmul is None:
            quantize_attention_matmul = _env_bool("PI05_WHOLE_QUANTIZE_ATTN_MATMUL", True)
        if quantize is None:
            quantize = not _env_bool("PI05_WHOLE_EXPORT_NO_QUANT", False)

        onnx_path = export_whole_model_to_onnx(
            self.pi05_model,
            export_dir,
            num_steps=num_steps,
            precision=str(precision),
            config_obj=config_obj if config_obj is not None else self._config_obj,
            checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else self._checkpoint_dir,
            num_calibration_samples=num_calibration_samples,
            enable_llm_nvfp4=enable_llm_nvfp4,
            quantize_attention_matmul=quantize_attention_matmul,
            quantize=quantize,
        )
        self.is_quantized = bool(quantize)
        self.pi05_model = self.pi05_model  # explicit for clarity
        return str(onnx_path.resolve())
