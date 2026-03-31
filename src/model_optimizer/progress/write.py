import os
import json
import time
from dataclasses import dataclass

from ..webui.extras.constants import PROGRESS_LOG, QUANTIZE_LOG, RUNNING_LOG


def _append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def write_running_log(log_dir, content):
    file_path = os.path.join(log_dir, RUNNING_LOG)
    os.makedirs(log_dir, exist_ok=True)
    with open(file_path, 'a+', encoding="utf-8") as f:
        f.write(content + '\n')

def write_quantize_progress(log_dir,
                   percentage, current_steps, total_steps, 
                   elapsed_time, remaining_time):
    content = {
        "percentage": percentage,
        "current_steps": current_steps,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "remaining_time": remaining_time
    }

    file_path = os.path.join(log_dir, QUANTIZE_LOG)
    _append_jsonl(file_path, content)


def write_progress(
    log_dir: str,
    *,
    percentage: float,
    current_steps: int,
    total_steps: int,
    elapsed_time: str,
    remaining_time: str,
    phase: str | None = None,
    step_name: str | None = None,
) -> None:
    """写入 WebUI 通用进度文件 ``progress.jsonl``。

    WebUI 侧 ``control.get_running_info`` 会读取该文件，并用最后一条记录更新进度条。
    字段保持兼容：percentage/current_steps/total_steps/elapsed_time/remaining_time。
    额外字段（phase/step_name）用于未来扩展“分步骤展示”，不会影响现有读取逻辑。
    """
    payload: dict = {
        "percentage": float(percentage),
        "current_steps": int(current_steps),
        "total_steps": int(total_steps),
        "elapsed_time": str(elapsed_time),
        "remaining_time": str(remaining_time),
    }
    if phase is not None:
        payload["phase"] = str(phase)
    if step_name is not None:
        payload["step_name"] = str(step_name)
    _append_jsonl(os.path.join(log_dir, PROGRESS_LOG), payload)


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class ProgressTracker:
    """轻量进度追踪器：把“步骤 i/N”转换成 WebUI 可展示的 percentage/ETA。"""

    log_dir: str
    total_steps: int
    start_ts: float = time.time()
    current_steps: int = 0

    def start(self, *, phase: str = "start", step_name: str = "开始") -> None:
        self.current_steps = 0
        self._emit(phase=phase, step_name=step_name)

    def advance(self, *, step_name: str, phase: str = "running", step_incr: int = 1) -> None:
        self.current_steps = min(self.total_steps, self.current_steps + int(step_incr))
        self._emit(phase=phase, step_name=step_name)

    def finish(self, *, ok: bool = True) -> None:
        self.current_steps = self.total_steps
        self._emit(phase="finished" if ok else "failed", step_name="完成" if ok else "失败")

    def _emit(self, *, phase: str, step_name: str) -> None:
        elapsed = time.time() - self.start_ts
        if self.total_steps <= 0:
            pct = 0.0
            remaining = 0.0
        else:
            pct = 100.0 * (self.current_steps / self.total_steps)
            if self.current_steps <= 0:
                remaining = 0.0
            else:
                rate = elapsed / max(1, self.current_steps)
                remaining = rate * (self.total_steps - self.current_steps)

        write_progress(
            self.log_dir,
            percentage=pct,
            current_steps=self.current_steps,
            total_steps=self.total_steps,
            elapsed_time=_format_hhmmss(elapsed),
            remaining_time=_format_hhmmss(remaining),
            phase=phase,
            step_name=step_name,
        )
