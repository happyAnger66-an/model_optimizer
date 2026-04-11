"""LeRobot 数据集加载与 openpi repack 辅助（从 eval_webui/dataset.py 提取）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training.data_loader import TransformedDataset

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError(
            "需要安装 lerobot（openpi 使用 lerobot.common.datasets.lerobot_dataset；"
            "新版包可能为 lerobot.datasets.lerobot_dataset）。"
        ) from e


def tree_to_numpy(obj: Any) -> Any:
    """LeRobot 常返回 torch.Tensor；openpi transforms 期望 numpy。"""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(tree_to_numpy(x) for x in obj)
    return obj


def build_repack_only(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose([*data_config.repack_transforms.inputs])


def unwrap_lerobot_base(dataset: Any) -> object:
    d = dataset
    while hasattr(d, "_dataset"):
        d = d._dataset
    return d


def global_episode_id_per_frame(base_ds: Any, n_frames: int) -> np.ndarray:
    try:
        hf = getattr(base_ds, "hf_dataset", None)
    except Exception as exc:
        logging.warning("读取 hf_dataset 失败: %s；episode 视为全 0。", exc)
        return np.zeros(n_frames, dtype=np.int64)
    if hf is not None:
        cols = getattr(hf, "column_names", []) or []
        for col in ("episode_index", "episode_idx", "ep_index"):
            if col in cols:
                ep = np.asarray(hf[col][:n_frames])
                if ep.shape[0] != n_frames:
                    raise RuntimeError(
                        f"episode 列 {col!r} 长度 {ep.shape[0]} 与 len(dataset)={n_frames} 不一致"
                    )
                return ep.astype(np.int64, copy=False)
    logging.warning(
        "未在 hf_dataset 中找到 episode_index；将整段当作 episode_id=0。"
    )
    return np.zeros(n_frames, dtype=np.int64)


def make_lerobot_dataset(
    *,
    repo_id: str,
    action_horizon: int,
    action_sequence_keys: tuple[str, ...],
    prompt_from_task: bool,
    dataset_root: Path | None,
):
    meta_kw: dict = {"repo_id": repo_id}
    if dataset_root is not None:
        meta_kw["root"] = str(dataset_root)
    try:
        meta = lerobot_dataset.LeRobotDatasetMetadata(**meta_kw)
    except TypeError:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    ds_kwargs: dict = {
        "repo_id": repo_id,
        "delta_timestamps": {
            key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys
        },
    }
    if dataset_root is not None:
        ds_kwargs["root"] = str(dataset_root)
    try:
        dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)
    except TypeError:
        ds_kwargs.pop("root", None)
        logging.warning("当前 lerobot 版本忽略 dataset_root，请用 HF 缓存或升级 lerobot。")
        dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)
    if prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset
