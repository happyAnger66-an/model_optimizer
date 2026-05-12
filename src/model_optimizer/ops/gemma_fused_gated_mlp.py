# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Pi05 PaliGemma **语言解码器** FFN：合并 Gate+Up 线性 + GeGLU + Down（仅 bf16/fp16）。

数学与 OpenPI ``GemmaMLP`` 等价::

    down_proj(act_fn(gate_proj(x)) * up_proj(x))

其中 ``act_fn`` 由 ``config.hidden_act`` 决定；Pi05 文本塔为 ``gelu_pytorch_tanh``。

**仅替换** ``LLM.construct_model`` 所持有的 ``paligemma.get_decoder()``，不修改 action expert。

导出 ONNX 为单节点：设置环境变量 ``MODEL_OPTIMIZER_GEMMA_FUSED_MLP_TRT_EXPORT=1`` 并在
``LLM.export`` 前注册 :func:`gemma_fused_gated_mlp_plugin.register_gemma_fused_gated_mlp_onnx_symbolic_functions`。
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _encode_hidden_act(hidden_act: str) -> int:
    """与 ``siglip_mlp`` 对齐的稳定 ``act_id``（ONNX / TRT 插件属性）。"""

    name_to_id = {
        "gelu": 0,
        "gelu_pytorch_tanh": 1,
        "relu": 2,
        "silu": 3,
        "swish": 3,
        "quick_gelu": 4,
        "gelu_new": 5,
    }
    return int(name_to_id.get(hidden_act, -1))


def _act_fn_from_id(aid: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if aid == 0 or aid == 1:
        return lambda t: F.gelu(t, approximate="tanh")
    if aid == 2:
        return F.relu
    if aid == 3:
        return F.silu
    if aid == 4:

        def _qg(t: torch.Tensor) -> torch.Tensor:
            return t * torch.sigmoid(1.702 * t)

        return _qg
    if aid == 5:
        return lambda t: F.gelu(t, approximate="none")
    return lambda t: F.gelu(t, approximate="tanh")


def gemma_fused_gated_mlp_eager(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_id: int,
) -> torch.Tensor:
    """参考实现：``gate_up_weight`` 形状 ``[2*inter, hidden]``，``down_weight`` ``[hidden, inter]``。"""

    fn = _act_fn_from_id(int(act_id))
    z = F.linear(x, gate_up_weight, bias=None)
    inter = z.shape[-1] // 2
    gate, up = z.split(inter, dim=-1)
    h = fn(gate) * up
    return F.linear(h, down_weight, bias=None)


class FusedGemmaMLP(nn.Module):
    """与 ``GemmaMLP`` 等价的融合实现（单路 ``gate_up`` 权重）。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        try:
            from openpi.models_pytorch.transformers_replace.models.gemma.modeling_gemma import (
                ACT2FN,
            )

            self.act_fn = ACT2FN[config.hidden_act]
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenPI Gemma ACT2FN import failed (%s); using torch gelu tanh.", exc)
            self.act_fn = lambda t: F.gelu(t, approximate="tanh")
        self._act_id = _encode_hidden_act(str(config.hidden_act))

    @classmethod
    def from_gemma_mlp(cls, mlp: nn.Module) -> "FusedGemmaMLP":
        """从现有 ``GemmaMLP``（或结构兼容模块）拷贝权重。"""
        cfg = mlp.config
        out = cls(cfg)
        with torch.no_grad():
            w_g = mlp.gate_proj.weight.data
            w_u = mlp.up_proj.weight.data
            out.gate_up.weight.data.copy_(torch.cat([w_g, w_u], dim=0))
            out.down_proj.weight.data.copy_(mlp.down_proj.weight.data)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_trt = os.environ.get("MODEL_OPTIMIZER_GEMMA_FUSED_MLP_TRT_EXPORT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if use_trt and self._act_id >= 0:
            from .gemma_fused_gated_mlp_plugin import gemma_fused_gated_mlp_plugin

            return gemma_fused_gated_mlp_plugin(
                x,
                self.gate_up.weight,
                self.down_proj.weight,
                self._act_id,
            )
        z = self.gate_up(x)
        gate, up = z.split(self.intermediate_size, dim=-1)
        h = self.act_fn(gate) * up
        return self.down_proj(h)


def patch_decoder_fused_gated_mlp(decoder: nn.Module, *, enabled: bool) -> int:
    """将解码器各层 ``GemmaMLP`` 换为 :class:`FusedGemmaMLP`（仅 ``hidden_act == gelu_pytorch_tanh``）。"""

    if not enabled:
        return 0
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return 0
    try:
        from transformers.models.gemma.modeling_gemma import (
            GemmaMLP,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("patch_decoder_fused_gated_mlp: cannot import GemmaMLP (%s)", exc)
        return 0

    n = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not isinstance(mlp, GemmaMLP):
            continue
        if str(getattr(mlp.config, "hidden_act", "")) != "gelu_pytorch_tanh":
            logger.debug("Skip fused MLP for hidden_act=%r", getattr(mlp.config, "hidden_act", None))
            continue
        layer.mlp = FusedGemmaMLP.from_gemma_mlp(mlp)
        n += 1
    if n:
        logger.info("FusedGemmaMLP: patched %d decoder layer(s) (gelu_pytorch_tanh only).", n)
    return n
