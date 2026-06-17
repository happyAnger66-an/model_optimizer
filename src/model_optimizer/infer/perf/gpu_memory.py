"""CUDA 显存分阶段采样（PyTorch allocator 统计，供 webui 加载与 executor 打点）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from termcolor import colored

logger = logging.getLogger(__name__)

_active_profiler: GpuMemoryProfiler | None = None


@dataclass
class GpuMemorySnapshot:
    tag: str
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    peak_reserved_gb: float


@dataclass
class GpuMemoryProfiler:
    """分阶段记录 ``memory_allocated`` / ``max_memory_allocated`` 峰值。"""

    enabled: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    snapshots: list[GpuMemorySnapshot] = field(default_factory=list)
    after_infer_reported: bool = False

    @classmethod
    def resolve_device(cls, device: str | None) -> torch.device:
        if device:
            dev = torch.device(device)
            if dev.type == "cuda":
                return dev
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def available(self) -> bool:
        return bool(self.enabled and self.device.type == "cuda" and torch.cuda.is_available())

    def reset_peak(self) -> None:
        if not self.available():
            return
        torch.cuda.reset_peak_memory_stats(self.device)

    def report(self, tag: str, *, print_summary: bool = False) -> GpuMemorySnapshot | None:
        if not self.available():
            return None
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
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
                f"peak_reserved={snap.peak_reserved_gb:.2f}GB",
                "cyan",
            ),
            flush=True,
        )
        if print_summary:
            print(torch.cuda.memory_summary(self.device, abbreviated=True), flush=True)
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


def start_gpu_mem_profile(*, enabled: bool, device: str | None = None) -> GpuMemoryProfiler:
    global _active_profiler
    _active_profiler = GpuMemoryProfiler(
        enabled=bool(enabled),
        device=GpuMemoryProfiler.resolve_device(device),
    )
    if _active_profiler.available():
        _active_profiler.reset_peak()
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
