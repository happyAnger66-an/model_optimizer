# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""在 **ONNX 导出** 阶段把 Gemma **eager GQA attention** 映射为 ONNX ``Attention`` 单算子。

设计目标（与 ``LLM.export(..., mode=\"self_forward\")`` / **仅 prefix** 对齐）：

- 运行时数学尽量与 ``eager_attention_forward`` 一致：用 ``torch.nn.functional.scaled_dot_product_attention``
  处理 **GQA**（``Q`` 为 ``[B, Hq, S, D]``，``K/V`` 为 ``[B, Hkv, S, D]``）+ **加性 mask**（与 eager 中 ``causal_mask`` 切片一致）。
- ONNX 侧通过 ``torch.autograd.Function.symbolic`` 发出 ``Attention`` + ``Transpose``，便于 TensorRT
  ``IAttention`` 导入（具体是否走融合内核仍由 TRT 版本/图结构决定）。

启用方式（导出前设置环境变量）::

    export MODEL_OPTIMIZER_GEMMA_ONNX_ATTENTION_EXPORT=1
    # 可选：ONNX opset（默认 23；若 PyTorch 脚本导出告警，可降到 20）
    export MODEL_OPTIMIZER_GEMMA_ONNX_OPSET=23

注意：

- ``attention_mask is None`` 或 ``kwargs[\"output_attentions\"]`` 为真时，**自动回退**到原始 ``eager_attention_forward``，
  以免与 HF 行为/可视化权重需求不一致。
- 该补丁通过替换 **已加载** 的 ``modeling_gemma`` 模块中的 ``eager_attention_forward`` 全局函数实现；
  导出结束务必调用 :func:`unpatch_gemma_eager_attention_for_onnx_attention`。
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ORIGINAL_EAGER_ATTENTION_FORWARD: Optional[Callable[..., Any]] = None
_PATCHED_MODULE: Optional[str] = None


class GemmaOnnxGqaAttentionFunction(torch.autograd.Function):
    """GQA attention：forward 用 SDPA；symbolic 导出为 ONNX ``Attention``。"""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        scaling: float,
        dropout_p: float,
    ) -> torch.Tensor:
        # SDPA 对 GQA 的语义与 Gemma eager_attention_forward（repeat_kv + matmul）在 fp32 softmax 路径上对齐良好。
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=False,
            scale=float(scaling),
        )
        # 与 eager_attention_forward 一致：返回 [B, S, Hq, D]
        return out.transpose(1, 2).contiguous()

    @staticmethod
    def symbolic(  # type: ignore[override]
        g: torch._C.Graph,
        query: torch._C.Value,
        key: torch._C.Value,
        value: torch._C.Value,
        attn_mask: torch._C.Value,
        scaling: torch._C.Value,
        dropout_p: torch._C.Value,
    ) -> torch._C.Value:
        # scaling/dropout_p 仅用于 PyTorch 前向；ONNX Attention 默认 scale = 1/sqrt(head_size)。
        # 若未来需要显式 scale，可在此读取常量并写入 scale_f（需与 torch 侧一致）。
        del scaling, dropout_p
        attn = g.op("Attention", query, key, value, attn_mask, is_causal_i=0)
        return g.op("Transpose", attn, perm_i=[0, 2, 1, 3])


def _eager_attention_forward_onnx_export(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        raise RuntimeError("gemma_onnx_attention_export: original eager_attention_forward is not captured")

    if kwargs.get("output_attentions", False):
        return _ORIGINAL_EAGER_ATTENTION_FORWARD(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    if attention_mask is None:
        return _ORIGINAL_EAGER_ATTENTION_FORWARD(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    # 与 modeling_gemma.eager_attention_forward 的切片规则保持一致
    causal_mask = attention_mask[:, :, :, : key.shape[-2]]
    dropout_p = float(dropout) if module.training else 0.0
    attn_out = GemmaOnnxGqaAttentionFunction.apply(
        query,
        key,
        value,
        causal_mask,
        float(scaling),
        dropout_p,
    )
    return attn_out, None


def patch_gemma_eager_attention_for_onnx_attention(model: torch.nn.Module) -> None:
    """将 ``model`` 所在包内的 ``eager_attention_forward`` 替换为 ONNX 导出路径。"""
    global _ORIGINAL_EAGER_ATTENTION_FORWARD, _PATCHED_MODULE
    if not hasattr(model, "layers") or len(model.layers) == 0:
        raise ValueError("patch_gemma_eager_attention_for_onnx_attention: model has no decoder layers")
    mod_name = type(model.layers[0].self_attn).__module__
    mg = importlib.import_module(mod_name)
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        if not hasattr(mg, "eager_attention_forward"):
            raise AttributeError(f"{mod_name} has no eager_attention_forward; cannot patch")
        _ORIGINAL_EAGER_ATTENTION_FORWARD = mg.eager_attention_forward
        _PATCHED_MODULE = mod_name
        mg.eager_attention_forward = _eager_attention_forward_onnx_export
        logger.info("Patched %s.eager_attention_forward for ONNX Attention export", mod_name)
        return
    if _PATCHED_MODULE != mod_name:
        logger.warning(
            "gemma_onnx_attention_export: already patched module %r; skip re-patch for %r",
            _PATCHED_MODULE,
            mod_name,
        )


def unpatch_gemma_eager_attention_for_onnx_attention(model: torch.nn.Module) -> None:
    """恢复 ``eager_attention_forward``。"""
    global _ORIGINAL_EAGER_ATTENTION_FORWARD, _PATCHED_MODULE
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None or _PATCHED_MODULE is None:
        return
    if not hasattr(model, "layers") or len(model.layers) == 0:
        return
    mod_name = type(model.layers[0].self_attn).__module__
    if mod_name != _PATCHED_MODULE:
        logger.warning(
            "gemma_onnx_attention_export: unpatch module mismatch (patched=%r, got=%r); still restoring patched module",
            _PATCHED_MODULE,
            mod_name,
        )
    mg = importlib.import_module(_PATCHED_MODULE)
    mg.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION_FORWARD
    logger.info("Restored %s.eager_attention_forward", _PATCHED_MODULE)
    _ORIGINAL_EAGER_ATTENTION_FORWARD = None
    _PATCHED_MODULE = None


def onnx_attention_export_enabled() -> bool:
    v = os.environ.get("MODEL_OPTIMIZER_GEMMA_ONNX_ATTENTION_EXPORT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def onnx_attention_export_opset() -> int:
    return int(os.environ.get("MODEL_OPTIMIZER_GEMMA_ONNX_OPSET", "23"))
