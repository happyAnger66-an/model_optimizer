from __future__ import annotations

from ..model import Model
import contextlib
import os
import time
import torch

from termcolor import colored

import logging

from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.ops import patch_vision_siglip_mlp_custom_op
from model_optimizer.ops.siglip_ffn_fp8 import patch_vision_siglip_ffn_fp8_flashrt_custom_op
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

logger = logging.getLogger(__name__)


def _effective_siglip_mlp_custom_op(flag: bool | None) -> bool:
    """CLI/构造参数优先；未显式传入时可由环境变量 ``MODEL_OPTIMIZER_SIGLIP_MLP_CUSTOM_OP`` 开启。"""
    if flag is not None:
        return bool(flag)
    return os.environ.get("MODEL_OPTIMIZER_SIGLIP_MLP_CUSTOM_OP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _effective_siglip_ffn_fp8_flashrt_custom_op(flag: bool | None) -> bool:
    """构造参数优先；未传入时由 ``MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP`` 开启。"""
    if flag is not None:
        return bool(flag)
    return os.environ.get("MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _ffn_flashrt_trt_export_enabled() -> bool:
    """与 :class:`~model_optimizer.ops.siglip_ffn_fp8.SiglipFfFp8FlashrtMlpWrapper` 一致：导出 ONNX 为 ``trt::SiglipFfFp8FlashrtPlugin``。"""
    return os.environ.get("MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _maybe_register_siglip_ffn_fp8_flashrt_trt_export() -> None:
    """在 ``torch.onnx.export`` 前注册 ``trt::siglip_ffn_fp8_flashrt_plugin`` → ``trt::SiglipFfFp8FlashrtPlugin`` symbolic。"""
    if not _ffn_flashrt_trt_export_enabled():
        return
    from model_optimizer.ops.siglip_ffn_fp8_flashrt_export import (
        register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions,
    )

    register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions()


def _vit_onnx_opset_version() -> int:
    """FlashRT FP8 导出路径使用 FLOAT8 / 更高 opset，默认 20；其余保持 19。"""
    return 20 if _ffn_flashrt_trt_export_enabled() else 19


def _maybe_register_siglip_mlp_trt_export(vit: "Vit") -> None:
    """与 TensorRT-Edge-LLM 一致：在 ONNX 导出前注册 ``trt::SiglipMlpPlugin`` symbolic。"""
    if not getattr(vit, "siglip_mlp_custom_op", False):
        return
    if os.environ.get("MODEL_OPTIMIZER_SIGLIP_MLP_TRT_EXPORT", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    import model_optimizer.ops.siglip_mlp_plugin as _siglip_plugin  # noqa: F401 — registers custom op

    from model_optimizer.ops.siglip_mlp_plugin import (
        register_siglip_mlp_plugin_onnx_symbolic_functions,
    )

    register_siglip_mlp_plugin_onnx_symbolic_functions()


@contextlib.contextmanager
def _sdp_math_backend_only():
    """Force SDPA math path so TorchScript ONNX tracing avoids ops that hit ComplexDouble."""
    try:
        from torch.nn.attention import SDPBackend
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except Exception:
        pass
    torch_cuda = getattr(torch.backends, "cuda", None)
    sdp_kernel_fn = getattr(torch_cuda, "sdp_kernel", None) if torch_cuda is not None else None
    if sdp_kernel_fn is not None:
        with sdp_kernel_fn(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            yield
    else:
        yield


@contextlib.contextmanager
def _force_vision_eager_attention_temporarily(vision_tower: torch.nn.Module):
    """SigLIP blocks: use eager attention (matmul) during ONNX export (avoids SDPA export issues)."""
    cfg = getattr(vision_tower, "config", None)
    if cfg is None or not hasattr(cfg, "_attn_implementation"):
        yield
        return
    saved = getattr(cfg, "_attn_implementation", None)
    try:
        setattr(cfg, "_attn_implementation", "eager")
        yield
    finally:
        if saved is not None:
            setattr(cfg, "_attn_implementation", saved)


class Vit(torch.nn.Module, Model):
    """Pi05 视觉塔封装。

    SigLIP encoder MLP 融合：构造参数 ``siglip_mlp_custom_op=True`` 或环境变量
    ``MODEL_OPTIMIZER_SIGLIP_MLP_CUSTOM_OP=1`` 启用（见 :mod:`model_optimizer.ops.siglip_mlp`），
    不修改上游 ``modeling_siglip.py``。

    ONNX 导出为 ``trt::SiglipMlpPlugin``：同时开启上述融合与
    ``MODEL_OPTIMIZER_SIGLIP_MLP_TRT_EXPORT=1``；导出前注册 symbolic（类比
    ``onnx_export/llm_export.py`` 中对 ``register_attention_plugin_onnx_symbolic_functions`` 的调用）。
    前向在 :class:`~model_optimizer.ops.siglip_mlp.SiglipMlpCustomOpWrapper` 内直接调用
    ``siglip_mlp_plugin(...)``，与 ``layers.py`` 直接调用 ``attention_plugin(...)`` 一致。

    **FFN FP8 + FlashRT（TensorRT）**：构造 ``siglip_ffn_fp8_flashrt_custom_op=True`` 或环境变量
    ``MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP=1`` 为各层 ``mlp`` 包
    :class:`~model_optimizer.ops.siglip_ffn_fp8.SiglipFfFp8FlashrtMlpWrapper`；量化后导出 ONNX 时再设
    ``MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT=1``（与 ``export`` / ``quantize`` 末尾导出一致）。
    该路径与 ``siglip_mlp_custom_op`` 互斥（优先 FlashRT 包装）。
    """

    def __init__(
        self,
        config,
        vision_tower,
        multi_modal_projector,
        *,
        siglip_mlp_custom_op: bool | None = None,
        siglip_ffn_fp8_flashrt_custom_op: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.vision_tower = vision_tower
        self.model = vision_tower
        self.device = self.vision_tower.device
        self.multi_modal_projector = multi_modal_projector
        self.siglip_ffn_fp8_flashrt_custom_op = _effective_siglip_ffn_fp8_flashrt_custom_op(
            siglip_ffn_fp8_flashrt_custom_op
        )
        eff_mlp = _effective_siglip_mlp_custom_op(siglip_mlp_custom_op)
        if self.siglip_ffn_fp8_flashrt_custom_op and eff_mlp:
            logger.warning(
                "Vit: siglip_ffn_fp8_flashrt_custom_op and siglip_mlp_custom_op both requested; "
                "using FlashRT FFN wrapper only (SiglipMlpPlugin path disabled)."
            )
        self.siglip_mlp_custom_op = bool(eff_mlp) and not self.siglip_ffn_fp8_flashrt_custom_op
        if self.siglip_ffn_fp8_flashrt_custom_op:
            patch_vision_siglip_ffn_fp8_flashrt_custom_op(self.vision_tower, enabled=True)
        elif self.siglip_mlp_custom_op:
            patch_vision_siglip_mlp_custom_op(self.vision_tower, enabled=True)

    def get_calibrate_dataset(self, calib_data):
        return open_pi05_calib_for_quantize(calib_data, component="pi05_vit")

    def forward(self, pixel_values):
        #        logger.info(f'Pi05Vit input: {pixel_values.shape}')
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
#        print(colored(f"hidden_size: {self.config.text_config.hidden_size}", "green"))
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
#        logger.info(f'Pi05Vit output: {image_features.shape}')
        return image_features

    def export(self, export_dir, dynamo=True, mode=None):
        self.eval().cuda()

        pixel_values = torch.randn(
            (1, 3, 224, 224), dtype=torch.float32, device="cuda")

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start Vit export onnx...", "green"))
        _maybe_register_siglip_ffn_fp8_flashrt_trt_export()
        _maybe_register_siglip_mlp_trt_export(self)
        _opset = _vit_onnx_opset_version()
        with torch.inference_mode():
            with _force_vision_eager_attention_temporarily(self.vision_tower):
                with _sdp_math_backend_only():
                    torch.onnx.export(
                        self,
                        (pixel_values),  # Include position_ids in ONNX export
                        f"{output_dir}/vit.onnx",
                        # Add position_ids to input names
                        input_names=["pixel_values"],
                        output_names=["image_features"],
                        opset_version=_opset,
                        dynamo=dynamo,
                        do_constant_folding=True,
                        #                dynamic_axes={
                        #                    "pixel_values": {0: "batch_size"},
                        #                    "vit_embeds": {0: "batch_size"},
                        #                },
                    )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(f"Vit export onnx done to {output_dir} cost:{end - start}s", "green"))
        return self

    @classmethod
    def construct_from_name_path(
        cls,
        model_name,
        model_path,
        train_config=None,
        *,
        siglip_mlp_custom_op: bool | None = None,
        siglip_ffn_fp8_flashrt_custom_op: bool | None = None,
    ):
        from .model_pi05 import Pi05Model

        pi05_model = Pi05Model.construct_from_name_path(
            model_name, model_path, train_config
        )
        return cls.construct_model(
            pi05_model,
            siglip_mlp_custom_op=siglip_mlp_custom_op,
            siglip_ffn_fp8_flashrt_custom_op=siglip_ffn_fp8_flashrt_custom_op,
        )

    @classmethod
    def construct_model(
        cls,
        pi05_model,
        dtype=torch.bfloat16,
        *,
        siglip_mlp_custom_op: bool | None = None,
        siglip_ffn_fp8_flashrt_custom_op: bool | None = None,
    ):
        eff_flash = _effective_siglip_ffn_fp8_flashrt_custom_op(siglip_ffn_fp8_flashrt_custom_op)
        eff_mlp = _effective_siglip_mlp_custom_op(siglip_mlp_custom_op)
        vit_model = cls(
            pi05_model.paligemma_with_expert.paligemma.config,
            pi05_model.paligemma_with_expert.paligemma.model.vision_tower,
            pi05_model.paligemma_with_expert.paligemma.model.multi_modal_projector,
            siglip_mlp_custom_op=eff_mlp,
            siglip_ffn_fp8_flashrt_custom_op=eff_flash,
        )
        # pi05_model.paligemma_with_expert.paligemma.model.multi_modal_projector).to(dtype)
        return vit_model

    @classmethod
    def export_onnx(cls, pi05_model, export_dir):
        vit_model = cls.construct_model(pi05_model, dtype=torch.bfloat16)
        vit_model.eval().cuda()

        pixel_values = torch.randn(
            (1, 3, 224, 224), dtype=torch.bfloat16, device="cuda")

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        _maybe_register_siglip_ffn_fp8_flashrt_trt_export()
        _maybe_register_siglip_mlp_trt_export(vit_model)
        _opset = _vit_onnx_opset_version()
        with torch.inference_mode():
            with _force_vision_eager_attention_temporarily(vit_model.vision_tower):
                with _sdp_math_backend_only():
                    torch.onnx.export(
                        vit_model,
                        (pixel_values),  # Include position_ids in ONNX export
                        f"{output_dir}/vit.onnx",
                        # Add position_ids to input names
                        input_names=["pixel_values"],
                        output_names=["vit_embeds"],
                        opset_version=_opset,
                        dynamo=False,
                        do_constant_folding=True,
                        dynamic_axes={
                            "pixel_values": {0: "batch_size"},
                            "vit_embeds": {0: "batch_size"},
                        },
                    )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return vit_model

    def quantize(self, quant_cfg, calib_data, export_dir, *, measure_quant_error: bool = False):
        # tokenizer = get_tokenizer(model_dir)
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        from model_optimizer.quantization.quantization_utils import quantize_model  # noqa: F401
        quantize_model(
            self, quant_cfg, calib_dataloader, measure_quant_error=measure_quant_error
        )
        self.is_quantized = True
        _dyn = (
            "fp16"
            if _ffn_flashrt_trt_export_enabled() and is_nvfp4_quantized(quant_cfg)
            else "bf16"
        )
        set_dynamic_quant(self, _dyn)

        self.export(export_dir, dynamo=False)
        onnx_path = f"{export_dir}/vit.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)