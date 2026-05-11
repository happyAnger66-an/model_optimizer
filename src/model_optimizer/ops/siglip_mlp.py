# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP 编码器 MLP 的融合替换（方案 A），**不修改** ``modeling_siglip.py``。

在运行时把 ``vision_tower.vision_model.encoder.layers[*].mlp`` 换成
:class:`SiglipMlpCustomOpWrapper`，前向与 ``SiglipMLP`` 数值等价。

- 默认：:func:`siglip_mlp_eager`（纯 ``torch.nn.functional`` 融合）。
- 导出 ONNX 为 TRT 单节点：设置 ``MODEL_OPTIMIZER_SIGLIP_MLP_TRT_EXPORT=1``，
  ``Vit.export`` 前注册 symbolic；前向**直接调用** :func:`~model_optimizer.ops.siglip_mlp_plugin.siglip_mlp_plugin`
  （与 Edge-LLM ``layers.py`` 里 ``attention_plugin(...)`` 用法一致，不经 ``torch.ops`` 再包一层）。
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
    """Map HuggingFace ``hidden_act`` to stable ``act_id`` for fusion dispatch."""

    try:
        from openpi.models_pytorch.transformers_replace.models.siglip.modeling_siglip import (
            ACT2FN,
        )

        if hidden_act in ACT2FN:
            name_to_id = {
                "gelu": 0,
                "gelu_pytorch_tanh": 1,
                "relu": 2,
                "silu": 3,
                "swish": 3,
                "quick_gelu": 4,
                "gelu_new": 5,
            }
            return int(name_to_id.get(hidden_act, 0))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not import openpi Siglip ACT2FN (%s); trying transformers.", exc)

    try:
        from transformers.models.siglip.modeling_siglip import ACT2FN as ACT2FN_T

        if hidden_act in ACT2FN_T:
            name_to_id = {
                "gelu": 0,
                "gelu_pytorch_tanh": 1,
                "relu": 2,
                "silu": 3,
                "swish": 3,
                "quick_gelu": 4,
                "gelu_new": 5,
            }
            return int(name_to_id.get(hidden_act, 0))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not import transformers Siglip ACT2FN (%s); using local fallbacks.", exc)

    fallback_map: dict[str, int] = {
        "gelu": 0,
        "gelu_pytorch_tanh": 1,
        "relu": 2,
        "silu": 3,
        "swish": 3,
        "quick_gelu": 4,
    }
    return int(fallback_map.get(hidden_act, -1))


def _siglip_mlp_reference(
    x: torch.Tensor,
    fc1_w: torch.Tensor,
    fc1_b: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_b: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    h = F.linear(x, fc1_w, fc1_b)
    h = act_fn(h)
    return F.linear(h, fc2_w, fc2_b)


def _act_fn_from_id(aid: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if aid == 0 or aid == 1:
        return lambda t: F.gelu(t, approximate="tanh")
    if aid == 2:
        return F.relu
    if aid == 3:
        return F.silu
    if aid == 4:
        return lambda t: t * torch.sigmoid(1.702 * t)
    if aid == 5:
        return lambda t: F.gelu(t, approximate="none")
    return lambda t: F.gelu(t, approximate="tanh")


def siglip_mlp_eager(
    x: torch.Tensor,
    fc1_w: torch.Tensor,
    fc1_b: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_b: torch.Tensor,
    act_id: int,
) -> torch.Tensor:
    """与 ``SiglipMLP`` 等价的融合前向（Linear → act → Linear），供包装层与测试使用。"""

    fn = _act_fn_from_id(int(act_id))
    return _siglip_mlp_reference(x, fc1_w, fc1_b, fc2_w, fc2_b, fn)


class SiglipMlpCustomOpWrapper(nn.Module):
    """包装 HF ``SiglipMLP``：默认 :func:`siglip_mlp_eager`；TRT 导出时直接 :func:`~model_optimizer.ops.siglip_mlp_plugin.siglip_mlp_plugin`。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        cfg = getattr(inner, "config", None)
        hidden_act = getattr(cfg, "hidden_act", "gelu") if cfg is not None else "gelu"
        self._act_id = _encode_hidden_act(str(hidden_act))
        if self._act_id < 0:
            logger.warning(
                "Unknown Siglip hidden_act %r; wrapper will use plain eager forward.",
                hidden_act,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._act_id < 0:
            return self.inner(hidden_states)

        mlp = self.inner
        w1, b1, w2, b2 = mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias
        use_trt = os.environ.get(
            "MODEL_OPTIMIZER_SIGLIP_MLP_TRT_EXPORT", ""
        ).strip().lower() in ("1", "true", "yes")
        if use_trt:
            # 与 ``layers.py`` 中 ``attention_plugin(...)`` 相同：直接调用 ``@torch.library.custom_op`` 装饰的函数。
            from .siglip_mlp_plugin import siglip_mlp_plugin

            return siglip_mlp_plugin(hidden_states, w1, b1, w2, b2, self._act_id)
        return siglip_mlp_eager(hidden_states, w1, b1, w2, b2, self._act_id)


def _iter_siglip_encoder_layers(vision_tower: nn.Module) -> list[nn.Module]:
    vm = getattr(vision_tower, "vision_model", None)
    if vm is None:
        return []
    enc = getattr(vm, "encoder", None)
    if enc is None:
        return []
    layers = getattr(enc, "layers", None)
    if layers is None:
        return []
    return list(layers)


def patch_vision_siglip_mlp_custom_op(vision_tower: nn.Module, *, enabled: bool) -> int:
    """将各 encoder layer 的 ``mlp`` 替换为 :class:`SiglipMlpCustomOpWrapper`（``enabled`` 时）。"""

    if not enabled:
        return 0

    n = 0
    for layer in _iter_siglip_encoder_layers(vision_tower):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if isinstance(mlp, SiglipMlpCustomOpWrapper):
            continue
        layer.mlp = SiglipMlpCustomOpWrapper(mlp)
        n += 1
    if n:
        logger.info("SigLIP MLP fusion wrapper: patched %d encoder layer(s).", n)
    else:
        logger.warning(
            "SigLIP MLP fusion enabled but no encoder layers found under vision_tower."
        )
    return n
