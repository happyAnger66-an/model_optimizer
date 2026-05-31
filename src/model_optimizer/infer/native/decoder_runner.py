from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .graph_capture import (
    NativeGraphEntry,
    build_graph_entry_for_denoise_step,
    _flatten_past_key_values,
)

logger = logging.getLogger(__name__)


def _signature_key_for_denoise(
    prefix_pad_masks: torch.Tensor,
    past_key_values: Any,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[Any, ...]:
    return (
        tuple(prefix_pad_masks.shape),
        prefix_pad_masks.dtype,
        tuple(x_t.shape),
        x_t.dtype,
        tuple(timestep.shape),
        timestep.dtype,
        tuple(
            (tuple(t.shape), t.dtype) for t in _flatten_past_key_values(past_key_values)
        ),
    )


class NativeDenoiseLoopRunner:
    """MVP：原生 denoise_step 运行器（可选 CUDA Graph）。"""

    def __init__(
        self,
        raw_denoise_step: Callable[..., torch.Tensor],
        *,
        use_cuda_graph: bool = True,
        graph_warmup: int = 3,
        perf: bool = False,
    ) -> None:
        self._raw = raw_denoise_step
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_warmup = max(int(graph_warmup), 0)
        self.perf = bool(perf)
        self._graph_cache: dict[tuple[Any, ...], NativeGraphEntry] = {}
        self._capture_ms: list[float] = []
        self._run_ms: list[float] = []

    def _maybe_build_entry(
        self,
        state: Any,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> NativeGraphEntry | None:
        key = _signature_key_for_denoise(prefix_pad_masks, past_key_values, x_t, timestep)
        entry = self._graph_cache.get(key)
        if entry is not None:
            return entry
        try:
            entry = build_graph_entry_for_denoise_step(
                self._raw,
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
                warmup=self.graph_warmup,
            )
            self._graph_cache[key] = entry
            self._capture_ms.append(entry.capture_ms)
            logger.info(
                "[native] captured denoise CUDA graph key=%s capture=%.2fms",
                str(key[:6]),
                entry.capture_ms,
            )
            return entry
        except Exception as exc:
            logger.warning("[native] capture failed, fallback eager: %s", exc)
            return None

    def run(
        self,
        state: Any,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        out = None
        if self.use_cuda_graph and torch.cuda.is_available():
            entry = self._maybe_build_entry(
                state, prefix_pad_masks, past_key_values, x_t, timestep
            )
            if entry is not None:
                out = entry.replay(
                    state, prefix_pad_masks, past_key_values, x_t, timestep
                )
        if out is None:
            out = self._raw(state, prefix_pad_masks, past_key_values, x_t, timestep)
        self._run_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    def dump_summary(self) -> str:
        lines = ["NativeDenoiseLoopRunner summary:"]
        if self._run_ms:
            a = np.asarray(self._run_ms, dtype=np.float64)
            lines.append(
                "  run_ms: "
                f"n={int(a.size)} mean={float(np.mean(a)):.3f} "
                f"p50={float(np.percentile(a, 50)):.3f} "
                f"p90={float(np.percentile(a, 90)):.3f} "
                f"p99={float(np.percentile(a, 99)):.3f}"
            )
        if self._capture_ms:
            c = np.asarray(self._capture_ms, dtype=np.float64)
            lines.append(
                "  capture_ms: "
                f"n={int(c.size)} mean={float(np.mean(c)):.3f} "
                f"p50={float(np.percentile(c, 50)):.3f}"
            )
        lines.append(f"  graph_cache_size={len(self._graph_cache)}")
        return "\n".join(lines)

