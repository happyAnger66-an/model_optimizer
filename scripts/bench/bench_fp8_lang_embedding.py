#!/usr/bin/env python3
"""Micro-benchmark FP16 vs FP8 language embedding lookup for pi0.5."""

from __future__ import annotations

import argparse
import statistics
import time
import tempfile

import torch
import torch.nn as nn

from model_optimizer.infer.kernels.fp8_lang_embedding import Fp8LangEmbeddingLookup
from model_optimizer.infer.perf.stage_perf import StagePerfCollector
from model_optimizer.quantization.fp8_lang_embedding import (
    load_lang_embedding_sidecar,
    save_lang_embedding_sidecar,
    sidecar_exists,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-dir", type=str, default=None)
    p.add_argument("--mode", choices=("fp16", "fp8", "both"), default="both")
    p.add_argument("--vocab-size", type=int, default=257152)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--lang-seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _make_random_weight(vocab: int, hidden: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(vocab, hidden, device=device, dtype=torch.float32)


def _bench_lookup(fn, token_ids: torch.Tensor, *, iters: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn(token_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(token_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def _print_stats(label: str, times_ms: list[float], meta_lines: list[str] | None = None) -> None:
    p90 = sorted(times_ms)[max(0, int(0.9 * len(times_ms)) - 1)]
    print(
        f"{label:<24} n={len(times_ms)} "
        f"mean={statistics.mean(times_ms):.4f} ms "
        f"p50={statistics.median(times_ms):.4f} ms "
        f"p90={p90:.4f} ms"
    )
    if meta_lines:
        for line in meta_lines:
            print(f"  {line}")


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    token_ids = torch.randint(
        0,
        min(args.vocab_size, 10000),
        (args.batch_size, args.lang_seq_len),
        device=device,
        dtype=torch.long,
    )

    if args.mode in ("fp16", "both"):
        weight = _make_random_weight(args.vocab_size, args.hidden_size, device)
        embed = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        embed.weight.data.copy_(weight.to(embed.weight.dtype))
        times = _bench_lookup(embed, token_ids, iters=args.iters, warmup=args.warmup, device=device)
        table_bytes = args.vocab_size * args.hidden_size * 2
        _print_stats("fp16", times, [f"table_bytes={table_bytes / (1024**2):.1f} MiB"])

    if args.mode in ("fp8", "both"):
        if args.engine_dir and sidecar_exists(args.engine_dir):
            tables = load_lang_embedding_sidecar(args.engine_dir, device=device)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                weight = _make_random_weight(args.vocab_size, args.hidden_size, torch.device("cpu"))
                save_lang_embedding_sidecar(weight, tmp)
                tables = load_lang_embedding_sidecar(tmp, device=device)

        lookup = Fp8LangEmbeddingLookup(
            tables,
            output_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        collector = StagePerfCollector(enabled=True)
        mask = torch.ones_like(token_ids, dtype=torch.bool)

        class _PWE:
            def embed_language_tokens(self, x):
                return lookup(x)

        pwe = _PWE()

        def _run(ids: torch.Tensor) -> torch.Tensor:
            from model_optimizer.infer.kernels.fp8_lang_embedding import embed_language_tokens_timed

            return embed_language_tokens_timed(pwe, ids, mask, collector, bytes_per_element=1)

        times = _bench_lookup(_run, token_ids, iters=args.iters, warmup=args.warmup, device=device)
        meta = lookup.memory_stats()
        _print_stats(
            "fp8",
            times,
            [
                f"table_bytes={meta.total_bytes / (1024**2):.1f} MiB "
                f"(saved {100.0 * meta.saved_bytes_vs_fp16() / meta.fp16_baseline_bytes():.1f}% vs fp16)",
            ],
        )
        for line in collector.format_meta_summary_lines():
            print(f"  {line}")


if __name__ == "__main__":
    main()
