#!/usr/bin/env python3
# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""AOT export entry for FMHA D=256 (π0.5 Gemma LLM prefill).

Invoked by kernelSrc/build_cutedsl.py with:
  --output_dir DIR --file_name NAME --function_prefix PREFIX --export_only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# kernelSrc/ must be on sys.path for fmha_d256_cutedsl imports.
_KERNEL_SRC = Path(__file__).resolve().parents[1]
if str(_KERNEL_SRC) not in sys.path:
    sys.path.insert(0, str(_KERNEL_SRC))

import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

from fmha_d256_cutedsl import fmha_helpers as fmha_utils
from fmha_d256_cutedsl import MixedInputFusedMultiHeadAttentionPrefillD256
from fmha_d256_cutedsl.host.tensor_layout import (
    mark_1d_dynamic,
    mark_bshd_dynamic,
    mark_kv_cache_dynamic,
)


def _parse_comma_separated_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(","))


def export_fmha_d256_llm_homo(
    *,
    q_shape: tuple[int, int, int, int],
    kv_cap: int,
    is_persistent: bool,
    is_causal: bool,
    bottom_right_align: bool,
    q_dtype,
    output_dir: str,
    file_name: str,
    function_prefix: str,
) -> None:
    """Compile LLM homo FMHA (BSHD Q/O + packed KV) and export_to_c."""
    tag = f"[{file_name}]"
    b, s_q, h_q, d = q_shape
    if d != 256:
        raise ValueError(f"head_dim must be 256, got {d}")

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    fmha_op = MixedInputFusedMultiHeadAttentionPrefillD256(
        scale_granularity=256,
        qk_acc_dtype=Float32,
        pv_acc_dtype=Float32,
        is_persistent=is_persistent,
        mask_type=mask_type,
        is_mixed_input=False,
    )

    import cutlass.torch as cutlass_torch

    stream = cutlass_torch.default_stream()
    h_k = 1  # representative GQA ratio; runtime uses dynamic shapes
    h_r = h_q // h_k
    cap = kv_cap

    cp_dtype = cp.float16
    q_cp = cp.zeros((b, s_q, h_q, d), dtype=cp_dtype)
    kv_cp = cp.zeros((b, 2, h_k, cap, d), dtype=cp_dtype)
    o_cp = cp.zeros((b, s_q, h_q, d), dtype=cp_dtype)

    def _to_cute(arr, element_type):
        t = from_dlpack(arr, assumed_align=16)
        t.element_type = element_type
        return t

    q_t = mark_bshd_dynamic(_to_cute(q_cp, q_dtype))
    kv_t = mark_kv_cache_dynamic(_to_cute(kv_cp, q_dtype))
    o_t = mark_bshd_dynamic(_to_cute(o_cp, q_dtype))

    s_k = s_q
    cu_kv_np = np.arange(b + 1, dtype=np.int32) * s_k
    cu_kv_cp = cp.asarray(cu_kv_np)
    cu_kv = mark_1d_dynamic(from_dlpack(cu_kv_cp, assumed_align=16))

    _wsl = Int32(0) if is_causal else Int32(0)
    scale_q = Float32(1.0)
    scale_k = Float32(1.0)
    scale_v = Float32(1.0)
    inv_scale_o = Float32(1.0)

    print(
        f"{tag} AOT compile: b={b}, s_q={s_q}, h_q={h_q}, h_k={h_k}, "
        f"cap={cap}, d={d}, persistent={is_persistent}, causal={is_causal}"
    )
    t0 = time.time()
    compiled = cute.compile(
        fmha_op.call_llm,
        q_t,
        kv_t,
        o_t,
        cu_kv,
        _wsl,
        scale_q,
        scale_k,
        scale_v,
        inv_scale_o,
        stream,
        options="--opt-level 2",
    )
    print(f"{tag} Compilation time: {time.time() - t0:.4f}s")

    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print(f"{tag} Exported to {output_dir}/{file_name}.{{h,o}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT export FMHA D=256 for model_optimizer")
    parser.add_argument("--q_shape", type=_parse_comma_separated_ints, default=(1, 128, 8, 256))
    parser.add_argument("--kv_cap", type=int, default=4096)
    parser.add_argument("--is_persistent", action="store_true")
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--bottom_right_align", action="store_true")
    parser.add_argument("--in_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./artifacts")
    parser.add_argument("--file_name", type=str, default="fmha_d256_homo_bf16")
    parser.add_argument("--function_prefix", type=str, default="fmha_d256_homo_bf16")
    args = parser.parse_args()

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required for CuTe DSL AOT export")

    if not args.export_only:
        parser.error("--export_only is required when invoked from build_cutedsl.py")

    export_fmha_d256_llm_homo(
        q_shape=args.q_shape,
        kv_cap=args.kv_cap,
        is_persistent=args.is_persistent,
        is_causal=args.is_causal,
        bottom_right_align=args.bottom_right_align,
        q_dtype=args.in_dtype,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )


if __name__ == "__main__":
    main()
