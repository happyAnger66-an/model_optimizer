#!/usr/bin/env python3
# Copyright 2026 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference benchmark for Gemma GeGLU MLP (same tensor shapes as TRT fused plugin).

Typical Pi05 text LLM (per layer, token-major):
  x:         [M, H]   with M = B*T (default B=1, T=968 -> M=968), H=2048
  gate_up:   [2*I, H] with I=16384 -> [32768, 2048]  (nn.Linear stores weight as [out,in])
  down:      [H, I]   -> [2048, 16384]

This matches ``gemma_fused_gated_mlp_eager`` / TRT plugin layout (gate_up row-major [2I,H], down [H,I]).

Run on Jetson Thor (CUDA):
  python3 scripts/bench_gemma_fused_mlp_torch.py --bf16 --iters 200 --warmup 20

Compare with C++ micro-bench (after building ``bench_gemma_fused_gated_mlp``):
  ./csrc/build/trt_plugins/gemma_fused_gated_mlp/bench_gemma_fused_gated_mlp
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.nn.functional as F


def _gelu_tanh_ref(g: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(g, approximate="tanh")


def bench_torch(
    *,
    m: int,
    h: int,
    inter: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    x = torch.randn(m, h, device=device, dtype=dtype) * 0.02
    w_gu = torch.randn(2 * inter, h, device=device, dtype=dtype) * 0.02
    w_d = torch.randn(h, inter, device=device, dtype=dtype) * 0.02

    def one() -> None:
        z = F.linear(x, w_gu, bias=None)
        gate, up = z.split(inter, dim=-1)
        h_mid = _gelu_tanh_ref(gate) * up
        _ = F.linear(h_mid, w_d, bias=None)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        one()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return float(statistics.mean(times_ms)), float(statistics.median(times_ms))


def main() -> None:
    p = argparse.ArgumentParser(description="Torch GeGLU MLP reference benchmark (Pi05-ish shapes).")
    p.add_argument("--b", type=int, default=int(os.environ.get("GEMMA_BENCH_B", "1")))
    p.add_argument("--t", type=int, default=int(os.environ.get("GEMMA_BENCH_T", "968")))
    p.add_argument("--h", type=int, default=int(os.environ.get("GEMMA_BENCH_H", "2048")))
    p.add_argument("--inter", type=int, default=int(os.environ.get("GEMMA_BENCH_INTER", "16384")))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (default if flag set).")
    p.add_argument("--fp16", action="store_true", help="Use float16.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark (use Jetson Thor or other CUDA GPU).")

    dev = torch.device("cuda", 0)
    if args.fp16 and args.bf16:
        raise SystemExit("Pass at most one of --bf16 / --fp16.")
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    m = int(args.b) * int(args.t)
    mean_ms, med_ms = bench_torch(
        m=m,
        h=int(args.h),
        inter=int(args.inter),
        device=dev,
        dtype=dtype,
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    i2 = int(args.inter) * 2
    flops_g1 = 2 * m * i2 * int(args.h)
    flops_g2 = 2 * m * int(args.h) * int(args.inter)
    flops = flops_g1 + flops_g2
    tflops_mean = (flops / (mean_ms * 1e-3)) / 1e12
    tflops_med = (flops / (med_ms * 1e-3)) / 1e12

    print(
        "torch_geglu_mlp_ref bench | "
        f"dtype={dtype} M={m} H={args.h} inter={args.inter} | "
        f"mean_ms={mean_ms:.4f} median_ms={med_ms:.4f} | "
        f"approx_gemm_tflops_mean={tflops_mean:.3f} approx_gemm_tflops_median={tflops_med:.3f} | "
        f"iters={args.iters} warmup={args.warmup}"
    )


if __name__ == "__main__":
    main()
