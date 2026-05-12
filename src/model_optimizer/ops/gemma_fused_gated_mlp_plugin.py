# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Gemma 融合 FFN：ONNX ``trt::GemmaFusedGatedMlp`` + ``torch.library.custom_op``。"""

from __future__ import annotations

import logging
from typing import Any

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

from .gemma_fused_gated_mlp import gemma_fused_gated_mlp_eager

logger = logging.getLogger(__name__)

ONNX_OPSET_VERSION = 19

_SCHEMA_REGISTERED = False
_SYMBOLIC_REGISTERED = False


def _register_onnx_schema() -> None:
    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    schema = OpSchema(
        name="GemmaFusedGatedMlp",
        domain="trt",
        since_version=ONNX_OPSET_VERSION,
        doc="Fused PaliGemma text FFN: Linear(gate+up) -> act(gate)*up -> down. bf16/fp16.",
        inputs=[
            OpSchema.FormalParameter(name="x", description="Hidden [*, H]", type_str="T"),
            OpSchema.FormalParameter(
                name="gate_up_weight",
                description="Fused gate+up [2*I, H]",
                type_str="T",
            ),
            OpSchema.FormalParameter(
                name="down_weight",
                description="Down proj [H, I]",
                type_str="T",
            ),
        ],
        outputs=[
            OpSchema.FormalParameter(name="y", description="Output [*, H]", type_str="T"),
        ],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)"],
                "I/O dtype (fp16/bf16 only for v1).",
            ),
        ],
        attributes=[
            OpSchema.Attribute(
                name="act_id",
                type=OpSchema.AttrType.INT,
                description="Activation id (see model_optimizer.ops.gemma_fused_gated_mlp)",
                required=True,
            ),
        ],
    )
    try:
        onnx.defs.register_schema(schema)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GemmaFusedGatedMlp schema not registered (likely duplicate): %s", exc)
    _SCHEMA_REGISTERED = True


@symbolic_helper.parse_args("v", "v", "v", "i")
def symbolic_gemma_fused_gated_mlp(
    g: Any,
    x: torch._C.Value,
    gate_up_w: torch._C.Value,
    down_w: torch._C.Value,
    act_id: int,
):
    y = g.op(
        "trt::GemmaFusedGatedMlp",
        x,
        gate_up_w,
        down_w,
        act_id_i=int(act_id),
        outputs=1,
    )
    y.setType(x.type())
    return y


@torch.library.custom_op("trt::gemma_fused_gated_mlp", mutates_args=())
def gemma_fused_gated_mlp_plugin(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_id: int,
) -> torch.Tensor:
    return gemma_fused_gated_mlp_eager(x, gate_up_weight, down_weight, int(act_id))


def register_gemma_fused_gated_mlp_onnx_symbolic_functions() -> None:
    global _SYMBOLIC_REGISTERED
    _register_onnx_schema()
    if _SYMBOLIC_REGISTERED:
        return
    register_custom_op_symbolic(
        "trt::gemma_fused_gated_mlp",
        symbolic_gemma_fused_gated_mlp,
        ONNX_OPSET_VERSION,
    )
    _SYMBOLIC_REGISTERED = True
    logger.info(
        "Registered ONNX symbolic for trt::gemma_fused_gated_mlp -> trt::GemmaFusedGatedMlp (opset %s)",
        ONNX_OPSET_VERSION,
    )
