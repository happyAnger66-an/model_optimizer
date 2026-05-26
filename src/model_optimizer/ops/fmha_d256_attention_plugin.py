# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""ONNX / torch.export helpers for trt::FmhaD256AttentionPlugin."""

from __future__ import annotations

from typing import Any

import torch
from torch.onnx import symbolic_helper

ONNX_OPSET = 19
_PLUGIN_REGISTERED = False


def gemma_attention_head_dims(native_attn: torch.nn.Module) -> tuple[int, int, int]:
    """``(num_q_heads, num_kv_heads, head_dim)`` for HF ``GemmaAttention`` / OpenPI 变体。"""
    cfg = getattr(native_attn, "config", None)
    if cfg is not None:
        num_q_heads = int(cfg.num_attention_heads)
        num_kv_heads = int(cfg.num_key_value_heads)
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = int(cfg.hidden_size) // num_q_heads
        else:
            head_dim = int(head_dim)
        return num_q_heads, num_kv_heads, head_dim

    num_q_heads = getattr(native_attn, "num_heads", None)
    num_kv_heads = getattr(native_attn, "num_key_value_heads", None)
    head_dim = getattr(native_attn, "head_dim", None)
    if num_q_heads is None or num_kv_heads is None or head_dim is None:
        raise AttributeError(
            f"Cannot resolve attention head dims from {type(native_attn)!r}; "
            "expected .config (Gemma) or .num_heads/.num_key_value_heads/.head_dim"
        )
    return int(num_q_heads), int(num_kv_heads), int(head_dim)


def _register_torch_op() -> None:
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return

    @torch.library.custom_op("model_opt::fmha_d256_attention", mutates_args=())
    def fmha_d256_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_fp16: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager dummy — real compute in TRT plugin."""
        out = q.new_zeros(q.shape)
        return out, kv_cache.clone()

    @fmha_d256_attention.register_fake
    def _fmha_d256_attention_fake(
        q, kv_cache, cu_kv_seqlens, num_q_heads, num_kv_heads, head_dim, use_fp16
    ):
        return q.new_zeros(q.shape), kv_cache.clone()

    _PLUGIN_REGISTERED = True


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "i")
def fmha_d256_attention_symbolic(
    g,
    q,
    kv_cache,
    cu_kv_seqlens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    use_fp16,
):
    return g.op(
        "trt::FmhaD256AttentionPlugin",
        q,
        kv_cache,
        cu_kv_seqlens,
        num_q_heads_i=num_q_heads,
        num_kv_heads_i=num_kv_heads,
        head_dim_i=head_dim,
        use_fp16_i=use_fp16,
        outputs=2,
    )


def register_fmha_d256_onnx_symbolic() -> None:
    _register_torch_op()
    torch.onnx.register_custom_op_symbolic(
        "model_opt::fmha_d256_attention",
        fmha_d256_attention_symbolic,
        ONNX_OPSET,
    )


def fmha_d256_attention_custom_translation_table() -> dict[Any, Any]:
    _register_torch_op()
    return {torch.ops.model_opt.fmha_d256_attention.default: _dynamo_onnx_fmha_d256}


def _dynamo_onnx_fmha_d256(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    q, kv_cache, cu_kv_seqlens, num_q_heads, num_kv_heads, head_dim, use_fp16 = args
    from onnxscript.values import Opset

    trt = Opset("trt", ONNX_OPSET)
    return trt.FmhaD256AttentionPlugin(
        q,
        kv_cache,
        cu_kv_seqlens,
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        use_fp16=int(use_fp16),
        _outputs=2,
    )


class FmhaD256Attention(torch.nn.Module):
    """π0.5 导出用 Attention：PyTorch 侧 dummy，ONNX 侧 FmhaD256AttentionPlugin。"""

    def __init__(
        self,
        native_attn: torch.nn.Module,
        *,
        use_fp16: bool = True,
    ) -> None:
        super().__init__()
        self.native = native_attn
        self.use_fp16 = use_fp16
        num_q_heads, num_kv_heads, head_dim = gemma_attention_head_dims(native_attn)
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        _register_torch_op()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.native
        bsz, seq_len, _ = hidden_states.shape
        head_dim = self._head_dim
        num_q_heads = self._num_q_heads
        num_kv_heads = self._num_kv_heads

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # BSHD for plugin: [B, S, H, D]
        q_bshd = q.transpose(1, 2).contiguous()
        # Packed KV cache for the CuTe DSL LLM ABI: [B, 2, H_kv, S, D].
        #
        # The plugin consumes K/V from this packed cache directly. Passing the
        # placeholder ``past_key_value`` here would make the kernel read random
        # trtexec input/output memory instead of the freshly projected K/V,
        # which can lead to invalid attention data or kernel-side deadlock.
        del past_key_value
        kv_cache = torch.stack((k, v), dim=1).contiguous()
        use_fp16 = 1 if self.use_fp16 else 0

        attn_bshd, kv_cache_out = torch.ops.model_opt.fmha_d256_attention(
            q_bshd,
            kv_cache,
            cu_kv_seqlens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            use_fp16,
        )
        # Plugin 输出 [B, S, H, D]；与 HF Gemma 一致，经 o_proj 回到 hidden_size。
        attn_out = attn_bshd.reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)
        return attn_out, kv_cache_out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("native",):
                raise
            return getattr(self.native, name)
