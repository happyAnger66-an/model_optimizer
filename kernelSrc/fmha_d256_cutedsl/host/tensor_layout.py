"""Layout dynamism markers for LLM BSHD / KV-cache tensors."""

import cutlass.cute as cute


def mark_bshd_dynamic(tensor):
    """Mark a contiguous [B, S, H, D] tensor's first three dims as dynamic."""
    so = (0, 1, 2, 3)
    return (
        tensor.mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )


def mark_kv_cache_dynamic(tensor):
    """Mark [B, 2, H_kv, S, D] KV cache; K/V selector dim stays static."""
    so = (0, 1, 2, 3, 4)
    return (
        tensor.mark_layout_dynamic(leading_dim=4)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
        .mark_compact_shape_dynamic(mode=3, stride_order=so)
    )


def mark_1d_dynamic(tensor):
    """Mark a 1-D tensor (e.g. cum_seqlen_k) as dynamic."""
    return tensor.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )
