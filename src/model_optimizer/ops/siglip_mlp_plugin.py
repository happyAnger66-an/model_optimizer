# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP MLP 的 TRT 插件占位与 ONNX 导出（对齐 TensorRT-Edge-LLM ``attention_plugin``）。

与 ``llm_models/layers/layers.py`` 中 ``from .attention_plugin import attention_plugin`` 后
**直接** ``attention_plugin(...)`` 相同：业务代码应 **直接调用** :func:`siglip_mlp_plugin`，
不要经 ``torch.ops.trt.*`` 再包一层。

- ``onnx.defs.register_schema`` → ``trt::SiglipMlpPlugin``；
- ``@torch.library.custom_op("trt::siglip_mlp_plugin")`` → 可追踪的 Python 入口；
- ``register_custom_op_symbolic`` → ONNX 单节点。

实现委托 :func:`model_optimizer.ops.siglip_mlp.siglip_mlp_eager`；TensorRT 需提供解析
``trt::SiglipMlpPlugin`` 的 Plugin。
"""

from __future__ import annotations

import logging
from typing import Any

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

from .siglip_mlp import siglip_mlp_eager

logger = logging.getLogger(__name__)

ONNX_OPSET_VERSION = 19

_SCHEMA_REGISTERED = False
_SYMBOLIC_REGISTERED = False


def _register_onnx_schema() -> None:
    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    schema = OpSchema(
        name="SiglipMlpPlugin",
        domain="trt",
        since_version=ONNX_OPSET_VERSION,
        doc="Fused SigLIP encoder MLP: Linear → activation → Linear (TRT plugin).",
        inputs=[
            OpSchema.FormalParameter(
                name="x",
                description="Hidden states [*, D]",
                type_str="T",
            ),
            OpSchema.FormalParameter(
                name="fc1_weight",
                description="fc1 weight [H, D]",
                type_str="T",
            ),
            OpSchema.FormalParameter(
                name="fc1_bias",
                description="fc1 bias [H]",
                type_str="T",
            ),
            OpSchema.FormalParameter(
                name="fc2_weight",
                description="fc2 weight [D, H]",
                type_str="T",
            ),
            OpSchema.FormalParameter(
                name="fc2_bias",
                description="fc2 bias [D]",
                type_str="T",
            ),
        ],
        outputs=[
            OpSchema.FormalParameter(
                name="y",
                description="MLP output, same rank as x",
                type_str="T",
            ),
        ],
        type_constraints=[
            (
                "T",
                ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
                "I/O dtype.",
            ),
        ],
        attributes=[
            OpSchema.Attribute(
                name="act_id",
                type=OpSchema.AttrType.INT,
                description="Activation id (see model_optimizer.ops.siglip_mlp)",
                required=True,
            ),
        ],
    )
    try:
        onnx.defs.register_schema(schema)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SiglipMlpPlugin ONNX schema not registered (likely duplicate): %s", exc)
    _SCHEMA_REGISTERED = True


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i")
def symbolic_siglip_mlp_plugin(
    g: Any,
    x: torch._C.Value,
    fc1_w: torch._C.Value,
    fc1_b: torch._C.Value,
    fc2_w: torch._C.Value,
    fc2_b: torch._C.Value,
    act_id: int,
):
    y = g.op(
        "trt::SiglipMlpPlugin",
        x,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        act_id_i=int(act_id),
        outputs=1,
    )
    y.setType(x.type())
    return y


@torch.library.custom_op("trt::siglip_mlp_plugin", mutates_args=())
def siglip_mlp_plugin(
    x: torch.Tensor,
    fc1_w: torch.Tensor,
    fc1_b: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_b: torch.Tensor,
    act_id: int,
) -> torch.Tensor:
    """与 ``SiglipMLP`` 等价的融合 MLP；ONNX 中对应 ``trt::SiglipMlpPlugin``。"""

    return siglip_mlp_eager(x, fc1_w, fc1_b, fc2_w, fc2_b, int(act_id))


def register_siglip_mlp_plugin_onnx_symbolic_functions() -> None:
    """在 ``torch.onnx.export`` 之前调用一次，注册 ONNX symbolic（与 Edge-LLM 导出流程一致）。"""

    global _SYMBOLIC_REGISTERED
    _register_onnx_schema()
    if _SYMBOLIC_REGISTERED:
        return
    register_custom_op_symbolic(
        "trt::siglip_mlp_plugin",
        symbolic_siglip_mlp_plugin,
        ONNX_OPSET_VERSION,
    )
    _SYMBOLIC_REGISTERED = True
    logger.info(
        "Registered ONNX symbolic for trt::siglip_mlp_plugin -> trt::SiglipMlpPlugin (opset %s)",
        ONNX_OPSET_VERSION,
    )
