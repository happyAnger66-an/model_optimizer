# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""以 ONNX 描述 ``trt::SiglipFfFp8Plugin`` 单节点图（不经过 TorchScript FP8 导出）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)

_SCHEMA_REGISTERED = False


def register_siglip_ffn_fp8_onnx_schema() -> None:
    """注册 ``trt::SiglipFfFp8Plugin`` 的 ONNX schema（便于 checker / 工具链）。"""

    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    from onnx.defs import OpSchema

    schema = OpSchema(
        name="SiglipFfFp8Plugin",
        domain="trt",
        since_version=20,
        doc="Fused SigLIP encoder FFN FP8 (FlashRT-compatible).",
        inputs=[
            OpSchema.FormalParameter(
                name="x", description="FP8 activations [S, D]", type_str="tensor(float8e4m3fn)"
            ),
            OpSchema.FormalParameter(
                name="residual", description="Residual FP16 [S, D]", type_str="tensor(float16)"
            ),
            OpSchema.FormalParameter(
                name="up_w", description="Up weight FP8 [D, H]", type_str="tensor(float8e4m3fn)"
            ),
            OpSchema.FormalParameter(
                name="down_w", description="Down weight FP8 [H, D]", type_str="tensor(float8e4m3fn)"
            ),
            OpSchema.FormalParameter(name="up_b", description="Up bias FP16 [H]", type_str="tensor(float16)"),
            OpSchema.FormalParameter(name="down_b", description="Down bias FP16 [D]", type_str="tensor(float16)"),
            OpSchema.FormalParameter(
                name="unit_scale", description="Device scale scalar [1]", type_str="tensor(float)"
            ),
        ],
        outputs=[
            OpSchema.FormalParameter(name="y", description="Output FP16 [S, D]", type_str="tensor(float16)")
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
        logger.debug("SiglipFfFp8Plugin schema not registered: %s", exc)
    _SCHEMA_REGISTERED = True


def _fp8_tensor_to_uint8_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn, got {t.dtype}")
    return t.contiguous().view(torch.uint8).cpu().numpy()


def build_siglip_ffn_fp8_onnx_from_module(
    module: torch.nn.Module,
    *,
    s: int,
    onnx_path: str | Path,
    alpha_up: float = 1.0,
    alpha_down: float = 1.0,
) -> Path:
    """从 :class:`TinySiglipFfFp8Mlp` 等模块导出仅含 ``SiglipFfFp8Plugin`` 的 ONNX 图。

    权重与 bias 以 initializer 固化；运行时输入为 ``x``、``residual``。
    """

    register_siglip_ffn_fp8_onnx_schema()
    m = module
    d = int(m.d)  # type: ignore[attr-defined]
    h = int(m.h)  # type: ignore[attr-defined]

    init_upw = numpy_helper.from_array(_fp8_tensor_to_uint8_numpy(m.up_w), name="up_w")  # type: ignore[attr-defined]
    init_dnw = numpy_helper.from_array(_fp8_tensor_to_uint8_numpy(m.down_w), name="down_w")  # type: ignore[attr-defined]
    init_upb = numpy_helper.from_array(m.up_b.detach().float().cpu().numpy().astype(np.float16), name="up_b")  # type: ignore[attr-defined]
    init_dnb = numpy_helper.from_array(m.down_b.detach().float().cpu().numpy().astype(np.float16), name="down_b")  # type: ignore[attr-defined]
    init_us = numpy_helper.from_array(
        m.unit_scale.detach().float().cpu().numpy().astype(np.float32), name="unit_scale"  # type: ignore[attr-defined]
    )

    vi_x = helper.make_tensor_value_info("x", TensorProto.FLOAT8E4M3FN, [s, d])
    vi_res = helper.make_tensor_value_info("residual", TensorProto.FLOAT16, [s, d])
    vo = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [s, d])

    node = helper.make_node(
        "SiglipFfFp8Plugin",
        inputs=["x", "residual", "up_w", "down_w", "up_b", "down_b", "unit_scale"],
        outputs=["y"],
        domain="trt",
    )
    node.attribute.extend(
        [
            helper.make_attribute("alpha_up", float(alpha_up)),
            helper.make_attribute("alpha_down", float(alpha_down)),
        ]
    )

    graph = helper.make_graph(
        [node],
        "siglip_ffn_fp8",
        [vi_x, vi_res],
        [vo],
        initializer=[init_upw, init_dnw, init_upb, init_dnb, init_us],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("trt", 1)],
    )
    onnx_path = Path(onnx_path)
    onnx.save(model, str(onnx_path))
    return onnx_path
