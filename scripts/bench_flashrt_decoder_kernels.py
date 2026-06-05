#!/usr/bin/env python3
"""Benchmark FlashRT decoder FVK kernels with synthetic production-shaped input.

Example:
    python scripts/bench_flashrt_decoder_kernels.py \
        --build-dir /path/to/FlashRT/build \
        --enc-seq 818 --iters 20 --warmup 5 \
        --json /tmp/flashrt_kernel_bench.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_optimizer.infer.native.flashrt_decoder.benchmark import (  # noqa: E402
    format_summary_table,
    run_synthetic_trace_benchmark,
    write_benchmark_json,
)

logger = logging.getLogger("bench_flashrt_decoder_kernels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace benchmark for fvk kernels used by flashrt_decoder.pipeline.decoder_forward",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--build-dir", default=None, help="FlashRT build dir containing flash_rt_kernels*.so")
    parser.add_argument("--fmha-so", default=None, help="Optional libfmha_fp16_strided.so path")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--warmup", type=int, default=3, help="Full decoder warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Measured full decoder iterations")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for synthetic tensors")

    parser.add_argument("--S", type=int, default=10, help="Action suffix token count")
    parser.add_argument("--D", type=int, default=1024, help="Decoder hidden size")
    parser.add_argument("--H", type=int, default=4096, help="MLP intermediate size")
    parser.add_argument("--layers", type=int, default=18, help="Decoder layer count")
    parser.add_argument("--steps", type=int, default=10, help="Denoise step count")
    parser.add_argument("--enc-seq", type=int, default=818, help="Prefix KV sequence length")
    parser.add_argument("--NH", type=int, default=8, help="Query head count")
    parser.add_argument("--HD", type=int, default=256, help="Head dimension")

    parser.add_argument("--json", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    logger.info(
        "FlashRT decoder kernel benchmark: S=%d D=%d H=%d layers=%d steps=%d enc_seq=%d NH=%d HD=%d warmup=%d iters=%d",
        args.S,
        args.D,
        args.H,
        args.layers,
        args.steps,
        args.enc_seq,
        args.NH,
        args.HD,
        args.warmup,
        args.iters,
    )

    state, recorder = run_synthetic_trace_benchmark(
        build_dir=args.build_dir,
        fmha_so=args.fmha_so,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
        S=args.S,
        D=args.D,
        H=args.H,
        layers=args.layers,
        steps=args.steps,
        enc_seq=args.enc_seq,
        NH=args.NH,
        HD=args.HD,
        seed=args.seed,
    )
    summaries = recorder.summaries()
    print()
    print(format_summary_table(summaries))
    total_ms = sum(x.total_ms for x in summaries)
    print(f"\nmeasured_kernel_total_ms={total_ms:.2f}  samples={len(recorder.samples)}  dims={state.dims}")

    if args.json:
        write_benchmark_json(args.json, state=state, recorder=recorder)
        logger.info("JSON written: %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
