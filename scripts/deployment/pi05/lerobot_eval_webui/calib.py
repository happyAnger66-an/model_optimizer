"""Pi0.5 PyTorch 校准数据收集（与 standalone_inference_script 一致）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from termcolor import colored


def unwrap_pytorch_pi05_model(policy: Any) -> Any | None:
    """取底层 Pi0.5 torch 模块供 calib hook 使用。"""
    inner = getattr(policy, "_policy", None)
    if inner is not None:
        m = getattr(inner, "_model", None)
        if m is not None:
            return m
    return getattr(policy, "_model", None)


def start_pi05_calib_collectors(
    policy: Any,
    save_dir: Path,
    *,
    max_samples: int = 0,
    calib_item: str = "all",
) -> list[Any]:
    from model_optimizer.calibrate.collector.pi05 import (
        Pi05DenoiseCalibCollector,
        Pi05ExpertCalibCollector,
        Pi05LLMCalibCollector,
        Pi05VitCalibCollector,
    )

    torch_model = unwrap_pytorch_pi05_model(policy)
    if torch_model is None:
        raise RuntimeError("无法从 policy 解析 _model（无 _policy._model / _model），无法挂 Pi0.5 calib。")
    save_str = str(save_dir.expanduser().resolve())
    item = str(calib_item).strip().lower()
    if item not in ("all", "vit", "llm", "expert", "denoise"):
        raise ValueError(f"非法 calib_item={calib_item!r}（仅允许 all/vit/llm/expert/denoise）")

    collectors: list[Any] = []
    if item in ("all", "llm"):
        collectors.append(Pi05LLMCalibCollector(torch_model, save_str, max_samples=max_samples))
    if item in ("all", "expert"):
        collectors.append(Pi05ExpertCalibCollector(torch_model, save_str, max_samples=max_samples))
    if item in ("all", "vit"):
        collectors.append(Pi05VitCalibCollector(torch_model, save_str, max_samples=max_samples))
    if item in ("all", "denoise"):
        collectors.append(Pi05DenoiseCalibCollector(torch_model, save_str, max_samples=max_samples))
    print(colored(f"[infer] Pi0.5 calib 收集已启用 → {save_str}", "green"), flush=True)
    return collectors


def stop_pi05_calib_collectors(collectors: list[Any] | None) -> None:
    if not collectors:
        return
    for c in collectors:
        try:
            c.stop_collect()
        except Exception:  # pragma: no cover
            logging.exception("calib stop_collect 失败: %s", type(c).__name__)
