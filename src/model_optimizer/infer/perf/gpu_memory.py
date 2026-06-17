"""CUDA 显存分阶段采样（PyTorch allocator 统计，供 webui 加载与 executor 打点）。"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from termcolor import colored

logger = logging.getLogger(__name__)

_GPU_MEM_PROFILE_ENV = "MO_GPU_MEM_PROFILE"

_active_profiler: GpuMemoryProfiler | None = None


@dataclass
class GpuMemorySnapshot:
    tag: str
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    peak_reserved_gb: float


def parse_cuda_device_index(device: str | None) -> int:
    """从 ``cuda`` / ``cuda:1`` 解析 GPU 下标；未指定或非 CUDA 字符串时返回 0。"""
    if device is None:
        return 0
    s = str(device).strip().lower()
    if not s.startswith("cuda"):
        return 0
    if s == "cuda":
        return 0
    m = re.match(r"cuda\s*:\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1))
    return 0


def resolve_cuda_index(device: str | None) -> int | None:
    """解析可用的 CUDA 设备下标；不可用时返回 ``None``。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return None
    if device is not None:
        s = str(device).strip().lower()
        if s and not s.startswith("cuda"):
            return None
    idx = parse_cuda_device_index(device)
    if idx < 0 or idx >= torch.cuda.device_count():
        logger.warning(
            "gpu_mem_profile: 请求的 cuda:%s 无效（device_count=%d），跳过显存统计",
            idx,
            torch.cuda.device_count(),
        )
        return None
    return idx


def _ensure_cuda_context(cuda_index: int) -> None:
    """在尚未分配任何 GPU 张量前，PyTorch 需 ``set_device`` 才能调用 memory API。"""
    torch.cuda.set_device(cuda_index)


@dataclass
class GpuMemoryProfiler:
    """分阶段记录 ``memory_allocated`` / ``max_memory_allocated`` 峰值。"""

    enabled: bool = False
    cuda_index: int | None = None
    snapshots: list[GpuMemorySnapshot] = field(default_factory=list)
    after_infer_reported: bool = False

    def available(self) -> bool:
        return bool(
            self.enabled
            and self.cuda_index is not None
            and torch.cuda.is_available()
        )

    def reset_peak(self) -> None:
        if not self.available():
            return
        idx = int(self.cuda_index)
        try:
            _ensure_cuda_context(idx)
            torch.cuda.reset_peak_memory_stats(idx)
        except RuntimeError as exc:
            logger.warning("gpu_mem_profile: reset_peak_memory_stats(cuda:%s) 失败: %s", idx, exc)

    def report(self, tag: str, *, print_summary: bool = False) -> GpuMemorySnapshot | None:
        if not self.available():
            return None
        idx = int(self.cuda_index)
        try:
            _ensure_cuda_context(idx)
            torch.cuda.synchronize(idx)
            alloc = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            peak_alloc = torch.cuda.max_memory_allocated(idx)
            peak_reserved = torch.cuda.max_memory_reserved(idx)
        except RuntimeError as exc:
            logger.warning("gpu_mem_profile: report(%s) 失败: %s", tag, exc)
            return None
        snap = GpuMemorySnapshot(
            tag=tag,
            allocated_gb=alloc / 1024**3,
            reserved_gb=reserved / 1024**3,
            peak_allocated_gb=peak_alloc / 1024**3,
            peak_reserved_gb=peak_reserved / 1024**3,
        )
        self.snapshots.append(snap)
        print(
            colored(
                f"[MEM] {tag}: alloc={snap.allocated_gb:.2f}GB "
                f"reserved={snap.reserved_gb:.2f}GB "
                f"peak_alloc={snap.peak_allocated_gb:.2f}GB "
                f"peak_reserved={snap.peak_reserved_gb:.2f}GB "
                f"(cuda:{idx})",
                "cyan",
            ),
            flush=True,
        )
        if print_summary:
            try:
                print(torch.cuda.memory_summary(idx, abbreviated=True), flush=True)
            except RuntimeError as exc:
                logger.warning("gpu_mem_profile: memory_summary 失败: %s", exc)
        return snap

    def report_after_first_infer_once(self) -> GpuMemorySnapshot | None:
        if self.after_infer_reported:
            return None
        snap = self.report("after_first_infer", print_summary=True)
        if snap is not None:
            self.after_infer_reported = True
        return snap


def get_gpu_mem_profiler() -> GpuMemoryProfiler | None:
    return _active_profiler


def resolve_gpu_mem_profile_enabled(args: Any | None = None) -> bool:
    """YAML ``gpu_mem_profile`` 或环境变量 ``MO_GPU_MEM_PROFILE=1``。"""
    raw = os.getenv(_GPU_MEM_PROFILE_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if args is not None and bool(getattr(args, "gpu_mem_profile", False)):
        return True
    return False


def start_gpu_mem_profile(*, enabled: bool, device: str | None = None) -> GpuMemoryProfiler:
    global _active_profiler
    cuda_index = resolve_cuda_index(device) if enabled else None
    _active_profiler = GpuMemoryProfiler(enabled=bool(enabled), cuda_index=cuda_index)
    if _active_profiler.available():
        _active_profiler.reset_peak()
    elif enabled and cuda_index is None:
        logger.warning(
            "gpu_mem_profile 已开启但 CUDA 不可用或非 CUDA device=%r，跳过 [MEM] 打印",
            device,
        )
    return _active_profiler


def gpu_mem_report(tag: str, *, print_summary: bool = False) -> GpuMemorySnapshot | None:
    profiler = _active_profiler
    if profiler is None or not profiler.enabled:
        return None
    return profiler.report(tag, print_summary=print_summary)


def gpu_mem_report_after_first_infer_once() -> GpuMemorySnapshot | None:
    profiler = _active_profiler
    if profiler is None or not profiler.enabled:
        return None
    return profiler.report_after_first_infer_once()
