from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .graph_capture import (
    NativeFullLoopGraphEntry,
    NativeGraphEntry,
    _tensor_meta_signature,
    build_graph_entry_for_sample_actions,
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
        log_interval: int = 50,
    ) -> None:
        self._raw = raw_denoise_step
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_warmup = max(int(graph_warmup), 0)
        self.perf = bool(perf)
        self.log_interval = max(int(log_interval), 1)
        self._graph_cache: dict[tuple[Any, ...], NativeGraphEntry] = {}
        self._capture_blacklist: set[tuple[Any, ...]] = set()
        self._capture_disabled_reason: str | None = None
        self._capture_ms: list[float] = []
        self._run_ms: list[float] = []
        self._num_calls = 0
        self._num_replay = 0
        self._num_eager = 0

    def _maybe_build_entry(
        self,
        state: Any,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> NativeGraphEntry | None:
        key = _signature_key_for_denoise(prefix_pad_masks, past_key_values, x_t, timestep)
        if self._capture_disabled_reason is not None:
            return None
        if key in self._capture_blacklist:
            return None
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
            msg = str(exc)
            if "cudaErrorStreamCaptureInvalidated" in msg or "previous error during capture" in msg:
                self._capture_disabled_reason = msg
                logger.warning(
                    "[native] capture invalidated, disable cuda graph for this process; fallback eager. "
                    "reason=%s "
                    "(单步 denoise_step 不支持 CUDA Graph；请确认 native_use_cuda_graph 已走 full-loop，"
                    "或设置 native_full_loop_graph: true / trt_enable_stage_profile: false)",
                    msg,
                )
            else:
                logger.warning("[native] capture failed, fallback eager: %s", exc)
            self._capture_blacklist.add(key)
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
        self._num_calls += 1
        out = None
        mode = "pt_eager_fallback"
        if self.use_cuda_graph and torch.cuda.is_available():
            entry = self._maybe_build_entry(
                state, prefix_pad_masks, past_key_values, x_t, timestep
            )
            if entry is not None:
                out = entry.replay(
                    state, prefix_pad_masks, past_key_values, x_t, timestep
                )
                mode = "pt_cuda_graph_replay"
                self._num_replay += 1
        if out is None:
            out = self._raw(state, prefix_pad_masks, past_key_values, x_t, timestep)
            self._num_eager += 1
        self._run_ms.append((time.perf_counter() - t0) * 1000.0)
        if self.perf and (
            self._num_calls <= 5 or self._num_calls % self.log_interval == 0
        ):
            logger.info(
                "[native] denoise call=%d mode=%s replay=%d eager=%d cache=%d blacklist=%d",
                self._num_calls,
                mode,
                self._num_replay,
                self._num_eager,
                len(self._graph_cache),
                len(self._capture_blacklist),
            )
        return out

    def dump_summary(self) -> str:
        lines = ["NativeDenoiseLoopRunner summary:"]
        replay_ratio = (
            float(self._num_replay) / float(self._num_calls)
            if self._num_calls > 0
            else 0.0
        )
        lines.append(
            "  path_stats: "
            f"calls={self._num_calls} replay={self._num_replay} "
            f"eager={self._num_eager} replay_ratio={replay_ratio*100.0:.1f}%"
        )
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
        if self._capture_disabled_reason is not None:
            lines.append(f"  capture_disabled_reason={self._capture_disabled_reason}")
        return "\n".join(lines)


class NativeDenoiseLoopRunnerV2:
    """整段 num_steps 的 full-loop CUDA Graph 运行器。

    目标：把 ``sample_actions`` 内部 denoise 循环一次性 capture，推理阶段单次 replay。
    """

    def __init__(
        self,
        raw_sample_actions: Callable[..., torch.Tensor],
        *,
        use_cuda_graph: bool = True,
        graph_warmup: int = 3,
        default_num_steps: int = 10,
        perf: bool = False,
        log_interval: int = 20,
    ) -> None:
        self._raw = raw_sample_actions
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_warmup = max(int(graph_warmup), 0)
        self.default_num_steps = max(int(default_num_steps), 1)
        self.perf = bool(perf)
        self.log_interval = max(int(log_interval), 1)
        self._graph_cache: dict[tuple[Any, ...], NativeFullLoopGraphEntry] = {}
        self._capture_blacklist: set[tuple[Any, ...]] = set()
        self._capture_disabled_reason: str | None = None
        self._capture_ms: list[float] = []
        self._run_ms: list[float] = []
        self._num_calls = 0
        self._num_replay = 0
        self._num_eager = 0

    def _signature_key(
        self,
        *,
        device: torch.device | str,
        observation: Any,
        noise: torch.Tensor,
        num_steps: int,
    ) -> tuple[Any, ...]:
        return (
            str(device),
            int(num_steps),
            tuple(noise.shape),
            str(noise.dtype),
            str(noise.device),
            _tensor_meta_signature(observation),
        )

    def _maybe_build_entry(
        self,
        *,
        device: torch.device | str,
        observation: Any,
        noise: torch.Tensor,
        num_steps: int,
    ) -> NativeFullLoopGraphEntry | None:
        key = self._signature_key(
            device=device,
            observation=observation,
            noise=noise,
            num_steps=num_steps,
        )
        if self._capture_disabled_reason is not None:
            return None
        if key in self._capture_blacklist:
            return None
        entry = self._graph_cache.get(key)
        if entry is not None:
            return entry
        try:
            entry = build_graph_entry_for_sample_actions(
                self._raw,
                device=device,
                observation=observation,
                noise=noise,
                num_steps=int(num_steps),
                warmup=self.graph_warmup,
            )
            self._graph_cache[key] = entry
            self._capture_ms.append(entry.capture_ms)
            logger.info(
                "[native-v2] captured full-loop CUDA graph steps=%d capture=%.2fms",
                int(num_steps),
                entry.capture_ms,
            )
            return entry
        except Exception as exc:
            msg = str(exc)
            if (
                "cudaErrorStreamCaptureInvalidated" in msg
                or "previous error during capture" in msg
            ):
                self._capture_disabled_reason = msg
                logger.warning(
                    "[native-v2] capture invalidated, disable full-loop graph; fallback eager. reason=%s",
                    msg,
                )
            else:
                logger.warning("[native-v2] capture failed, fallback eager: %s", exc)
            self._capture_blacklist.add(key)
            return None

    def run(
        self,
        *,
        device: torch.device | str,
        observation: Any,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        n_steps = int(num_steps) if num_steps is not None else self.default_num_steps
        n_steps = max(n_steps, 1)
        if noise is None:
            # 首次调用必须显式传 noise 以建立图；后续若要支持隐式 noise 需接入模型 sample_noise。
            raise ValueError("NativeDenoiseLoopRunnerV2 requires explicit noise tensor")
        if not torch.is_tensor(noise):
            raise TypeError("noise must be Tensor")

        t0 = time.perf_counter()
        self._num_calls += 1
        out = None
        mode = "pt_eager_fallback"
        if self.use_cuda_graph and torch.cuda.is_available():
            entry = self._maybe_build_entry(
                device=device,
                observation=observation,
                noise=noise,
                num_steps=n_steps,
            )
            if entry is not None:
                out = entry.replay(observation=observation, noise=noise)
                mode = "pt_cuda_graph_full_loop_replay"
                self._num_replay += 1
        if out is None:
            out = self._raw(device, observation, noise=noise, num_steps=n_steps)
            self._num_eager += 1

        self._run_ms.append((time.perf_counter() - t0) * 1000.0)
        if self.perf and (
            self._num_calls <= 5 or self._num_calls % self.log_interval == 0
        ):
            logger.info(
                "[native-v2] sample_actions call=%d mode=%s replay=%d eager=%d cache=%d blacklist=%d",
                self._num_calls,
                mode,
                self._num_replay,
                self._num_eager,
                len(self._graph_cache),
                len(self._capture_blacklist),
            )
        return out

    def dump_summary(self) -> str:
        lines = ["NativeDenoiseLoopRunnerV2 summary:"]
        replay_ratio = (
            float(self._num_replay) / float(self._num_calls)
            if self._num_calls > 0
            else 0.0
        )
        lines.append(
            "  path_stats: "
            f"calls={self._num_calls} replay={self._num_replay} "
            f"eager={self._num_eager} replay_ratio={replay_ratio*100.0:.1f}%"
        )
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
        if self._capture_disabled_reason is not None:
            lines.append(f"  capture_disabled_reason={self._capture_disabled_reason}")
        return "\n".join(lines)

