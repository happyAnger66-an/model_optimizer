"""Pi0.5 Thor decoder 权重 repack（移植自 FlashRT 权重 spec/loader）。

把 HF/openpi 的 pi05 action-expert 权重重排成 FlashRT decoder kernel 期望的 FP8 布局，
填充 :class:`~model_optimizer.infer.native.flashrt_decoder.driver.DecoderWeights` 所需的
``_dec_{qkv,o,gu,d}_flat`` + ``_ae_w_scales`` + 单例（ain/aow/aob）。

对应 FlashRT：
- ``frontends/torch/_pi05_thor_spec.py::_decoder_block``（per-layer transform 序列）
- ``executors/torch_weights.py``（FusedQKV/FusedGateUp/tT/Quant/FlatCat）
- ``core/thor_frontend_utils.py``（``interleave_qk`` / ``quant_fp8``）

每层 transform（use_fp8=True）：
- qkv: FusedQKV(q[interleave 8 heads], k[interleave 1 head], v) → cat[2560,D].fp16 → ``.t()`` → FP8
- o:   o_proj.weight → fp16 → ``.t()`` → FP8
- gu:  cat[gate;up][2H,D].fp16 → ``.t()`` → FP8
- d:   down_proj.weight → fp16 → ``.t()`` → FP8
权重 scale 以 q→o→gu→d/层 的顺序追加进 ``_ae_w_scales``（``[layers*4]`` f32）。

与 ``pipeline.decoder_forward`` 的 GEMM 维度核对一致：
- qw_ptr 步进 ``D*2560``，weight=[K=D, N=2560]
- ow_ptr 步进 ``NH*HD*D``，weight=[K=NH*HD, N=D]
- gw_ptr 步进 ``D*H*2``，weight=[K=D, N=2H]
- dw_ptr 步进 ``H*D``，weight=[K=H, N=D]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

_FP16 = torch.float16
_FP8 = torch.float8_e4m3fn

# action_out_proj 默认扩散步数（-1/steps 烘进权重，与 FlashRT 一致）。
_DEFAULT_STEPS = 10
_DEC_PREFIX = "paligemma_with_expert.gemma_expert.model.layers.{i}"


def interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Q/K 输出维 pair-interleave（适配 RoPE kernel 布局）。移植自 thor_frontend_utils。"""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
        .reshape(num_heads, 2, head_dim // 2, in_dim)
        .permute(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """FP8 E4M3 per-tensor 量化。返回 ``(fp8_tensor, scale)``。移植自 thor_frontend_utils。"""
    w = w.contiguous()
    a = w.float().abs().max().item()
    s = max(a / 448.0, 1e-12)
    return (w.float() / s).clamp(-448, 448).to(_FP8), s


@dataclass
class RepackedDecoderWeights:
    """repack 产物：保活张量 + data_ptr 暴露给 driver.DecoderWeights。"""

    dec_qkv_flat: torch.Tensor
    dec_o_flat: torch.Tensor
    dec_gu_flat: torch.Tensor
    dec_d_flat: torch.Tensor
    ae_w_scales: torch.Tensor  # [layers*4] f32（device）
    ain_w: torch.Tensor
    ain_b: torch.Tensor
    aow: torch.Tensor
    aob: torch.Tensor
    num_layers: int
    use_fp8: bool
    _extra: dict = field(default_factory=dict)

    def ptr(self, name: str) -> int:
        return getattr(self, name).reshape(-1).data_ptr()


def _get_fp16(get: Callable[[str], torch.Tensor], key: str, device: str) -> torch.Tensor:
    t = get(key)
    return t.to(device=device, dtype=_FP16)


def repack_decoder_weights(
    get_weight: Callable[[str], torch.Tensor],
    *,
    num_layers: int = 18,
    num_q_heads: int = 8,
    steps: int = _DEFAULT_STEPS,
    use_fp8: bool = True,
    device: str = "cuda",
    layer_prefix: str = _DEC_PREFIX,
) -> RepackedDecoderWeights:
    """重排 decoder 权重。

    Args:
        get_weight: ``key -> Tensor`` 取权重（如来自 HF state_dict）。键名形如
            ``paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight``。
        num_q_heads: Q head 数（pi05=8；K/V 为 1 个 head）。
        steps: action_out_proj 的 -1/steps 缩放。
        use_fp8: True 走 FP8 量化布局；False 保持 fp16（baseline）。
    """
    qkv_parts: list[torch.Tensor] = []
    o_parts: list[torch.Tensor] = []
    gu_parts: list[torch.Tensor] = []
    d_parts: list[torch.Tensor] = []
    w_scales: list[float] = []

    def _maybe_quant(w: torch.Tensor) -> torch.Tensor:
        if use_fp8:
            fp8, s = quant_fp8(w)
            w_scales.append(float(s))
            return fp8
        return w

    for i in range(num_layers):
        dp = layer_prefix.format(i=i)

        # qkv: FusedQKV(interleave q=num_q_heads, k=1) → cat → fp16 → .t() → [FP8]
        q = get_weight(f"{dp}.self_attn.q_proj.weight").float()
        k = get_weight(f"{dp}.self_attn.k_proj.weight").float()
        v = get_weight(f"{dp}.self_attn.v_proj.weight").float()
        q = interleave_qk(q, num_q_heads)
        k = interleave_qk(k, 1)
        qkv = torch.cat([q, k, v], dim=0).to(device=device, dtype=_FP16)
        qkv = qkv.t().contiguous()
        qkv_parts.append(_maybe_quant(qkv).reshape(-1))

        # o: o_proj.weight → fp16 → .t() → [FP8]
        o = _get_fp16(get_weight, f"{dp}.self_attn.o_proj.weight", device).t().contiguous()
        o_parts.append(_maybe_quant(o).reshape(-1))

        # gu: cat([gate; up]) → fp16 → .t() → [FP8]
        gate = _get_fp16(get_weight, f"{dp}.mlp.gate_proj.weight", device)
        up = _get_fp16(get_weight, f"{dp}.mlp.up_proj.weight", device)
        gu = torch.cat([gate, up], dim=0).t().contiguous()
        gu_parts.append(_maybe_quant(gu).reshape(-1))

        # d: down_proj.weight → fp16 → .t() → [FP8]
        d = _get_fp16(get_weight, f"{dp}.mlp.down_proj.weight", device).t().contiguous()
        d_parts.append(_maybe_quant(d).reshape(-1))

    dec_qkv_flat = torch.cat(qkv_parts)
    dec_o_flat = torch.cat(o_parts)
    dec_gu_flat = torch.cat(gu_parts)
    dec_d_flat = torch.cat(d_parts)

    ae_w_scales = torch.tensor(
        w_scales if use_fp8 else [0.0] * (num_layers * 4),
        dtype=torch.float32,
        device=device,
    )

    # 单例（fp16，gmm_fp16 路径，不量化）。action_out_proj 烘入 -1/steps。
    ain_w = _get_fp16(get_weight, "action_in_proj.weight", device).t().contiguous()
    ain_b = _get_fp16(get_weight, "action_in_proj.bias", device)
    scale = -1.0 / float(steps)
    aow = (_get_fp16(get_weight, "action_out_proj.weight", device).t().contiguous() * scale).contiguous()
    aob = (_get_fp16(get_weight, "action_out_proj.bias", device) * scale).contiguous()

    return RepackedDecoderWeights(
        dec_qkv_flat=dec_qkv_flat,
        dec_o_flat=dec_o_flat,
        dec_gu_flat=dec_gu_flat,
        dec_d_flat=dec_d_flat,
        ae_w_scales=ae_w_scales,
        ain_w=ain_w,
        ain_b=ain_b,
        aow=aow,
        aob=aob,
        num_layers=num_layers,
        use_fp8=use_fp8,
    )


def state_dict_getter(
    state_dict: dict[str, torch.Tensor], *, strip_prefix: str = ""
) -> Callable[[str], torch.Tensor]:
    """从 HF state_dict 构造 ``get_weight``，可选去掉 lerobot 的 ``model.`` 前缀。"""

    def _get(key: str) -> torch.Tensor:
        k = (strip_prefix + key) if strip_prefix else key
        if k not in state_dict:
            raise KeyError(f"weight key not found: {k}")
        return state_dict[k]

    return _get
