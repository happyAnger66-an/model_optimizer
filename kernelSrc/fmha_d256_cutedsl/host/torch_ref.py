"""Torch reference and tensor helpers for d=256 mixed-input FMHA."""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def create_tensor(shape, dtype):
    """Allocate a random torch/cute tensor pair for testing."""
    import cutlass.torch as cutlass_torch

    f32_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
        shape,
        torch.float32,
        permute_order=None,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=cutlass.torch.RandomInitConfig(
            min_val=-2 if dtype.is_float or dtype.signed else 0, max_val=2
        ),
    )

    _, torch_tensor = cutlass_torch.cute_tensor_like(
        f32_torch_tensor,
        dtype,
        is_dynamic_layout=True,
        assumed_align=32,
    )

    cute_tensor = from_dlpack(torch_tensor, assumed_align=128)
    cute_tensor.element_type = dtype

    return f32_torch_tensor, cute_tensor, torch_tensor


def run_torch_fmha_homo(
    q, k, v, scale_softmax=1.0, scale_output=1.0, is_causal=False
):
    """Reference FMHA with homogeneous Q/K/V dtype (no quantization)."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    if not h_q == h_k:
        repeat_factor = h_q // h_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    batch = q.shape[0]
    ref_list = []
    for batch_idx in range(batch):
        q_i = q[batch_idx]
        k_i = k[batch_idx]
        v_i = v[batch_idx]
        s_i = torch.einsum("hqd,hkd->hqk", q_i, k_i) * scale_softmax
        s_q_len = q_i.shape[1]
        s_k_len = k_i.shape[1]
        if is_causal:
            q_coords = torch.arange(0, s_q_len).view(-1, 1)
            k_coords = torch.arange(0, s_k_len).view(1, -1)
            _mask = k_coords > q_coords + s_k_len - s_q_len
            s_i = s_i.masked_fill(_mask, -torch.inf)
        p_i = s_i.softmax(dim=-1)
        ref_i = torch.einsum("hqk,hkd->hqd", p_i, v_i) * scale_output
        ref_list.append(ref_i)
    return torch.stack(ref_list)


def run_torch_fmha(
    q, k, v, scale_k, scale_v, scale_softmax=1.0, scale_output=1.0, is_causal=False
):
    """Reference FMHA: BF16 Q, INT8 K/V dequantized with per-channel scales."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    if not h_q == h_k:
        repeat_factor = h_q // h_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        scale_k = scale_k.repeat_interleave(repeat_factor, dim=1)
        scale_v = scale_v.repeat_interleave(repeat_factor, dim=1)
    scale_k = (
        scale_k.unsqueeze(-1)
        .repeat(1, 1, 1, 1, k.shape[3] // scale_k.shape[3])
        .reshape(k.shape)
    )
    scale_v = (
        scale_v.unsqueeze(-1)
        .repeat(1, 1, 1, 1, v.shape[3] // scale_v.shape[3])
        .reshape(v.shape)
    )
    batch = q.shape[0]
    ref_list = []
    for batch_idx in range(batch):
        q_i = q[batch_idx]
        k_i = k[batch_idx]
        v_i = v[batch_idx]
        scale_k_i = scale_k[batch_idx]
        scale_v_i = scale_v[batch_idx]
        s_i = torch.einsum("hqd,hkd->hqk", q_i, k_i * scale_k_i) * scale_softmax
        s_q_len = q_i.shape[1]
        s_k_len = k_i.shape[1]
        if is_causal:
            q_coords = torch.arange(0, s_q_len).view(-1, 1)
            k_coords = torch.arange(0, s_k_len).view(1, -1)
            _mask = k_coords > q_coords + s_k_len - s_q_len
            s_i = s_i.masked_fill(_mask, -torch.inf)
        p_i = s_i.softmax(dim=-1)
        ref_i = torch.einsum("hqk,hkd->hqd", p_i, v_i * scale_v_i) * scale_output
        ref_list.append(ref_i)
    return torch.stack(ref_list)
