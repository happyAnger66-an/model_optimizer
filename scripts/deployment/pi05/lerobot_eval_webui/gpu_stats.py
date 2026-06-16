"""通过 ``nvidia-smi`` 采样 GPU 利用率（不增加 pip 依赖；无 NVIDIA 驱动时返回 None）。"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .protocol import event_to_json

if TYPE_CHECKING:
    from .config import Args

logger = logging.getLogger(__name__)


def parse_cuda_device_index(device: str | None) -> int:
    """从 ``cuda`` / ``cuda:1`` 等解析设备下标；非 CUDA 或未识别时返回 0。"""
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


def effective_gpu_index(args: Args) -> int:
    if args.gpu_device_index is not None:
        return max(0, int(args.gpu_device_index))
    return parse_cuda_device_index(args.device)


def sample_gpu_util(device_index: int = 0) -> dict[str, Any] | None:
    """返回 ``gpu_util_pct`` / ``mem_util_pct``（0–100），失败返回 None。"""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[0] if proc.stdout else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return None
        return {
            "gpu_util_pct": float(parts[0]),
            "mem_util_pct": float(parts[1]),
        }
    except FileNotFoundError:
        logger.debug("nvidia-smi not found; GPU stats disabled.")
        return None
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as exc:
        logger.debug("sample_gpu_util failed: %s", exc)
        return None


async def run_gpu_stats_loop(
    *,
    args: Args,
    run_id: str,
    pump_task: asyncio.Task[None],
    emit_async: Callable[..., Awaitable[None]],
) -> None:
    """周期性广播 GPU 利用率；``pump_task`` 结束时本协程退出。"""
    interval = float(args.gpu_stats_interval_sec)
    if interval <= 0:
        return
    interval = max(0.2, interval)
    dev_idx = int(effective_gpu_index(args))
    loop = asyncio.get_running_loop()

    while True:
        # 不能用 wait_for(pump_task, timeout=…)：超时会取消 pump_task，导致 drain 在 get() 上 CancelledError。
        done, _pending = await asyncio.wait((pump_task,), timeout=interval, return_when=asyncio.FIRST_COMPLETED)
        if pump_task in done:
            return
        try:
            stats = await loop.run_in_executor(None, sample_gpu_util, dev_idx)
        except Exception:  # pragma: no cover
            logging.exception("gpu sample run_in_executor failed")
            continue
        if stats is None:
            continue
        await emit_async(
            event_to_json(
                {
                    "type": "gpu_stats",
                    "run_id": run_id,
                    "device_index": dev_idx,
                    "gpu_util_pct": stats["gpu_util_pct"],
                    "mem_util_pct": stats["mem_util_pct"],
                }
            ),
            add_history=False,
        )
