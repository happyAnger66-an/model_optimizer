"""Decoder 的 per-prompt 预计算（RoPE 表 + AdaRMS 调制风格）。

移植自 FlashRT ``frontends/torch/pi05_thor.py`` 的 set_prompt 预计算段：
- RoPE 表（``_dec_rope``）：suffix 位置 [enc_seq, enc_seq+Sa) 的交错 cos/sin，``[Sa, 256]``。
- AdaRMS 风格（``_sa_all`` / ``_sf_all`` / ``_fs_all``）：把时间步嵌入经 time_mlp 后，
  逐层乘以 input/post-attn/final 的 modulation Dense，得到每步每层的调制量。

这些就是 ``pipeline.decoder_forward`` 里 ``sa``/``sf``/``fs`` 指针指向的 buffer。
``sa`` 索引：``si = (s*layers + l) * S * 3D``；``fs`` 索引：``fi = s * S * 3D``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

_FP16 = torch.float16
_DEC_PREFIX = "paligemma_with_expert.gemma_expert.model.layers.{i}"
_FINAL_NORM_KEY = "paligemma_with_expert.gemma_expert.model.norm.dense"


def build_dec_rope(
    enc_seq: int,
    Sa: int,
    *,
    device: str = "cuda",
    head_dim: int = 256,
    rope_base: float = 10000.0,
    max_pos: int = 1200,
) -> torch.Tensor:
    """构建 decoder suffix 的 RoPE 表 ``[Sa, head_dim]``（交错 cos/sin）。

    与 FlashRT 一致：``inv_freq = 1/base**(arange(0,HD,2)/HD)``，位置取 [enc_seq, enc_seq+Sa)，
    输出 256 维为 ``[cos0,sin0,cos1,sin1,...]`` 交错布局。
    """
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(max_pos, device=device)[:, None].float()
    kp = inv_freq[None, :] * pos  # [max_pos, HD/2]
    kc = torch.cos(kp).to(_FP16)
    ks = torch.sin(kp).to(_FP16)
    s = enc_seq
    return (
        torch.cat([kc[s : s + Sa, :, None], ks[s : s + Sa, :, None]], dim=2)
        .reshape(Sa, head_dim)
        .contiguous()
    )


def _silu(x: torch.Tensor) -> torch.Tensor:
    return (x * torch.sigmoid(x))


def _time_embedding(t_val: float, Da: int, device: str) -> torch.Tensor:
    """单个时间步的正弦嵌入 ``[Da]`` fp16（与 FlashRT period 调度一致）。"""
    t_tensor = torch.tensor([t_val], device=device, dtype=torch.float64)
    fraction = torch.linspace(0, 1, Da // 2, device=device, dtype=torch.float64)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling * t_tensor
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(_FP16)


@dataclass
class AdaRmsStyles:
    """AdaRMS 预计算结果（保活 + data_ptr）。"""

    sa_all: torch.Tensor  # [steps*layers*Sa, 3Da] fp16
    sf_all: torch.Tensor  # [steps*layers*Sa, 3Da] fp16
    fs_all: torch.Tensor  # [steps*Sa, 3Da] fp16


def precompute_adarms_styles(
    get_weight: Callable[[str], torch.Tensor],
    *,
    Sa: int,
    Da: int,
    num_layers: int,
    steps: int = 10,
    device: str = "cuda",
    layer_prefix: str = _DEC_PREFIX,
    final_norm_key: str = _FINAL_NORM_KEY,
) -> AdaRmsStyles:
    """预计算每步每层的 AdaRMS 调制风格（sa/sf/fs）。

    流程（移植自 FlashRT）：
      time_emb = silu(silu(t_emb @ Win + bin) @ Wout + bout) → expand 到 Sa
      sa[s,l] = time_emb @ attn_mod_w[l].T + attn_mod_b[l]
      sf[s,l] = time_emb @ ffn_mod_w[l].T  + ffn_mod_b[l]
      fs[s]   = time_emb @ final_mod_w.T   + final_mod_b
    """
    def g(key: str) -> torch.Tensor:
        return get_weight(key).to(device=device, dtype=_FP16)

    D3a = 3 * Da
    sa_all = torch.zeros(steps * num_layers * Sa, D3a, dtype=_FP16, device=device)
    sf_all = torch.zeros(steps * num_layers * Sa, D3a, dtype=_FP16, device=device)
    fs_all = torch.zeros(steps * Sa, D3a, dtype=_FP16, device=device)

    time_mlp_in_w = g("time_mlp_in.weight")
    time_mlp_in_b = g("time_mlp_in.bias")
    time_mlp_out_w = g("time_mlp_out.weight")
    time_mlp_out_b = g("time_mlp_out.bias")
    attn_mod_w = [g(f"{layer_prefix.format(i=l)}.input_layernorm.dense.weight") for l in range(num_layers)]
    attn_mod_b = [g(f"{layer_prefix.format(i=l)}.input_layernorm.dense.bias") for l in range(num_layers)]
    ffn_mod_w = [g(f"{layer_prefix.format(i=l)}.post_attention_layernorm.dense.weight") for l in range(num_layers)]
    ffn_mod_b = [g(f"{layer_prefix.format(i=l)}.post_attention_layernorm.dense.bias") for l in range(num_layers)]
    final_mod_w = g(f"{final_norm_key}.weight")
    final_mod_b = g(f"{final_norm_key}.bias")

    for step in range(steps):
        emb = _time_embedding(1.0 - step / steps, Da, device).unsqueeze(0)
        tmp = (emb @ time_mlp_in_w.t() + time_mlp_in_b.unsqueeze(0)).float()
        tmp = _silu(tmp).to(_FP16)
        tmp2 = (tmp @ time_mlp_out_w.t() + time_mlp_out_b.unsqueeze(0)).float()
        tmp2 = _silu(tmp2).to(_FP16)
        time_emb = tmp2.expand(Sa, -1).contiguous()
        for layer in range(num_layers):
            idx = (step * num_layers + layer) * Sa
            sa_all[idx : idx + Sa] = time_emb @ attn_mod_w[layer].t() + attn_mod_b[layer].unsqueeze(0)
            sf_all[idx : idx + Sa] = time_emb @ ffn_mod_w[layer].t() + ffn_mod_b[layer].unsqueeze(0)
        fidx = step * Sa
        fs_all[fidx : fidx + Sa] = time_emb @ final_mod_w.t() + final_mod_b.unsqueeze(0)

    return AdaRmsStyles(sa_all=sa_all, sf_all=sf_all, fs_all=fs_all)
