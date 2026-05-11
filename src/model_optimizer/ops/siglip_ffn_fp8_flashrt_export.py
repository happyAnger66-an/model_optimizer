# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP FFN FP8（FlashRT 语义）ONNX 导出：``trt::SiglipFfFp8FlashrtPlugin`` 与 ``torch.library.custom_op``。

与 ``csrc/trt_plugins/siglip_ffn_fp8/siglip_ffn_fp8_flashrt_plugin.cpp`` 中插件名一致；
前向参考实现委托 :func:`model_optimizer.ops.siglip_ffn_fp8.siglip_ffn_fp8_eager`。
"""

from __future__ import annotations

import logging
from typing import Any

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

from .siglip_ffn_fp8 import siglip_ffn_fp8_eager

logger = logging.getLogger(__name__)

# ONNX FLOAT8 与 TRT FP8 路径常用 opset ≥ 20
ONNX_OPSET_VERSION = 20

_SCHEMA_REGISTERED = False
_SYMBOLIC_REGISTERED = False


def _register_onnx_schema() -> None:
    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    schema = OpSchema(
        name="SiglipFfFp8FlashrtPlugin",
        domain="trt",
        since_version=ONNX_OPSET_VERSION,
        doc="Fused SigLIP encoder FFN FP8 (FlashRT GemmRunner + static quant), TRT SiglipFfFp8FlashrtPlugin.",
        inputs=[
            OpSchema.FormalParameter(name="x", description="FP8 activations [*, D]", type_str="tensor(float8e4m3fn)"),
            OpSchema.FormalParameter(name="residual", description="Residual FP16 [*, D]", type_str="tensor(float16)"),
            OpSchema.FormalParameter(name="up_w", description="Up weight FP8 [D, H]", type_str="tensor(float8e4m3fn)"),
            OpSchema.FormalParameter(name="down_w", description="Down weight FP8 [H, D]", type_str="tensor(float8e4m3fn)"),
            OpSchema.FormalParameter(name="up_b", description="Up bias FP16 [H]", type_str="tensor(float16)"),
            OpSchema.FormalParameter(name="down_b", description="Down bias FP16 [D]", type_str="tensor(float16)"),
            OpSchema.FormalParameter(
                name="unit_scale", description="Static quant descale scalar [1]", type_str="tensor(float)"
            ),
        ],
        outputs=[
            OpSchema.FormalParameter(name="y", description="Output FP16 [*, D]", type_str="tensor(float16)")
        ],
        type_constraints=[],
        attributes=[
            OpSchema.Attribute("alpha_up", OpSchema.AttrType.FLOAT, "Up GEMM alpha", required=False),
            OpSchema.Attribute("alpha_down", OpSchema.AttrType.FLOAT, "Down GEMM alpha", required=False),
        ],
    )
    try:
        onnx.defs.register_schema(schema)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SiglipFfFp8FlashrtPlugin ONNX schema not registered (likely duplicate): %s", exc)
    _SCHEMA_REGISTERED = True


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "f", "f")
def symbolic_siglip_ffn_fp8_flashrt_plugin(
    g: Any,
    x_fp8: torch._C.Value,
    residual: torch._C.Value,
    up_w: torch._C.Value,
    down_w: torch._C.Value,
    up_b: torch._C.Value,
    down_b: torch._C.Value,
    unit_scale: torch._C.Value,
    alpha_up: float,
    alpha_down: float,
):
    y = g.op(
        "trt::SiglipFfFp8FlashrtPlugin",
        x_fp8,
        residual,
        up_w,
        down_w,
        up_b,
        down_b,
        unit_scale,
        alpha_up=float(alpha_up),
        alpha_down=float(alpha_down),
        outputs=1,
    )
    y.setType(residual.type())
    return y


@torch.library.custom_op("trt::siglip_ffn_fp8_flashrt_plugin", mutates_args=())
def siglip_ffn_fp8_flashrt_plugin(
    x_fp8: torch.Tensor,
    residual: torch.Tensor,
    up_w_fp8: torch.Tensor,
    down_w_fp8: torch.Tensor,
    up_b: torch.Tensor,
    down_b: torch.Tensor,
    unit_scale: torch.Tensor,
    alpha_up: float,
    alpha_down: float,
) -> torch.Tensor:
    """与 TensorRT ``SiglipFfFp8FlashrtPlugin`` 对应的 7 输入 + 两 alpha；eager 对齐 :func:`siglip_ffn_fp8_eager`。"""

    return siglip_ffn_fp8_eager(
        x_fp8,
        residual,
        up_w_fp8,
        down_w_fp8,
        up_b,
        down_b,
        unit_scale,
        float(alpha_up),
        float(alpha_down),
    )


def register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions() -> None:
    """在 ``torch.onnx.export`` 之前调用一次（与 ``Vit.export`` / ``quantize`` 末尾导出一致）。"""

    global _SYMBOLIC_REGISTERED
    _register_onnx_schema()
    if _SYMBOLIC_REGISTERED:
        return
    for opset in (19, int(ONNX_OPSET_VERSION)):
        if opset < 19:
            continue
        register_custom_op_symbolic(
            "trt::siglip_ffn_fp8_flashrt_plugin",
            symbolic_siglip_ffn_fp8_flashrt_plugin,
            opset,
        )
    _SYMBOLIC_REGISTERED = True
    logger.info(
        "Registered ONNX symbolic for trt::siglip_ffn_fp8_flashrt_plugin -> trt::SiglipFfFp8FlashrtPlugin (opsets 19,%s)",
        ONNX_OPSET_VERSION,
    )
