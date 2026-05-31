from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Callable
from typing import Any

import torch


def _is_dynamic_cache_like(past_key_values: Any) -> bool:
    return hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache")


def _flatten_past_key_values(past_key_values: Any) -> list[torch.Tensor]:
    """把 KV cache 统一拍平成 tensor 列表。

    支持两种常见形态：
    - ``list/tuple[(k,v), ...]``
    - ``transformers.DynamicCache``（``key_cache/value_cache``）
    """
    flat: list[torch.Tensor] = []
    if _is_dynamic_cache_like(past_key_values):
        keys = getattr(past_key_values, "key_cache")
        vals = getattr(past_key_values, "value_cache")
        if len(keys) != len(vals):
            raise ValueError(
                f"DynamicCache key/value lengths mismatch: {len(keys)} vs {len(vals)}"
            )
        for i, (k, v) in enumerate(zip(keys, vals, strict=True)):
            if not (torch.is_tensor(k) and torch.is_tensor(v)):
                raise TypeError(
                    f"Unexpected DynamicCache[{i}] element types: "
                    f"{type(k).__name__}, {type(v).__name__}"
                )
            flat.append(k)
            flat.append(v)
        return flat

    for i, entry in enumerate(past_key_values):
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            raise TypeError(
                f"Unexpected past_key_values[{i}] type: {type(entry)}; "
                "expected tuple/list(key, value)"
            )
        k, v = entry[0], entry[1]
        if not (torch.is_tensor(k) and torch.is_tensor(v)):
            raise TypeError(
                f"Unexpected past_key_values[{i}] element types: "
                f"{type(k).__name__}, {type(v).__name__}"
            )
        flat.append(k)
        flat.append(v)
    return flat


def _build_static_past_key_values(
    template: Any, static_flat: list[torch.Tensor]
) -> Any:
    if _is_dynamic_cache_like(template):
        obj = copy.copy(template)
        keys: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        for i in range(0, len(static_flat), 2):
            keys.append(static_flat[i])
            vals.append(static_flat[i + 1])
        obj.key_cache = keys
        obj.value_cache = vals
        return obj

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    j = 0
    for _ in template:
        out.append((static_flat[j], static_flat[j + 1]))
        j += 2
    return out


@dataclasses.dataclass
class NativeGraphEntry:
    key: tuple[Any, ...]
    graph: torch.cuda.CUDAGraph
    static_state: Any
    static_prefix_pad_masks: torch.Tensor
    static_past_flat: list[torch.Tensor]
    static_x_t: torch.Tensor
    static_timestep: torch.Tensor
    static_output: torch.Tensor
    capture_ms: float

    def replay(
        self,
        state: Any,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_tensor(self.static_state) and torch.is_tensor(state):
            self.static_state.copy_(state, non_blocking=True)
        self.static_prefix_pad_masks.copy_(prefix_pad_masks, non_blocking=True)
        flat = _flatten_past_key_values(past_key_values)
        for dst, src in zip(self.static_past_flat, flat, strict=True):
            dst.copy_(src, non_blocking=True)
        self.static_x_t.copy_(x_t, non_blocking=True)
        self.static_timestep.copy_(timestep, non_blocking=True)
        self.graph.replay()
        # 复制一份，避免后续 replay 覆盖当前输出内容。
        return self.static_output.clone()


def build_graph_entry_for_denoise_step(
    raw_denoise_step: Callable[..., torch.Tensor],
    state: Any,
    prefix_pad_masks: torch.Tensor,
    past_key_values: Any,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    warmup: int = 3,
) -> NativeGraphEntry:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA graph capture requires CUDA")
    if not (
        torch.is_tensor(prefix_pad_masks)
        and torch.is_tensor(x_t)
        and torch.is_tensor(timestep)
    ):
        raise TypeError("prefix_pad_masks/x_t/timestep must be torch.Tensor")

    stream = torch.cuda.Stream(device=x_t.device)
    cur_stream = torch.cuda.current_stream(device=x_t.device)
    stream.wait_stream(cur_stream)

    with torch.cuda.stream(stream):
        static_state = state
        if torch.is_tensor(state):
            static_state = torch.empty_like(state)
            static_state.copy_(state, non_blocking=False)

        static_prefix = torch.empty_like(prefix_pad_masks)
        static_prefix.copy_(prefix_pad_masks, non_blocking=False)

        in_flat = _flatten_past_key_values(past_key_values)
        static_flat = [torch.empty_like(t) for t in in_flat]
        for dst, src in zip(static_flat, in_flat, strict=True):
            dst.copy_(src, non_blocking=False)
        static_past = _build_static_past_key_values(past_key_values, static_flat)

        static_x_t = torch.empty_like(x_t)
        static_x_t.copy_(x_t, non_blocking=False)
        static_timestep = torch.empty_like(timestep)
        static_timestep.copy_(timestep, non_blocking=False)

    for _ in range(max(int(warmup), 0)):
        with torch.cuda.stream(stream):
            out = raw_denoise_step(
                static_state, static_prefix, static_past, static_x_t, static_timestep
            )
            if not torch.is_tensor(out):
                raise TypeError(
                    f"denoise_step output must be Tensor, got {type(out).__name__}"
                )
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        static_output = raw_denoise_step(
            static_state, static_prefix, static_past, static_x_t, static_timestep
        )
    stream.synchronize()
    cur_stream.wait_stream(stream)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    key = (
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
    return NativeGraphEntry(
        key=key,
        graph=graph,
        static_state=static_state,
        static_prefix_pad_masks=static_prefix,
        static_past_flat=static_flat,
        static_x_t=static_x_t,
        static_timestep=static_timestep,
        static_output=static_output,
        capture_ms=float(capture_ms),
    )

