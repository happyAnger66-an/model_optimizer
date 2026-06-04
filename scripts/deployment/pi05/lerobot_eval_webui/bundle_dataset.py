"""数据集与 repack 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from termcolor import colored

from .bundle_common import BundleProgress, get_train_config
from .config import Args
from .dataset import build_repack_only, make_lerobot_dataset


@dataclass
class DatasetBundle:
    train_cfg: Any
    data_config: Any
    dataset: Any
    repack_fn: Any
    action_horizon: int
    action_dim: int


def load_dataset_bundle(args: Args, progress: BundleProgress) -> DatasetBundle:
    print(colored("[infer] get_config + data_config ...", "cyan"), flush=True)
    progress.emit("config", "读取训练配置与 data_config …")
    train_cfg = get_train_config(args)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_dim = int(getattr(train_cfg.model, "action_dim", 0) or 0)
    if action_dim <= 0:
        raise ValueError("train_cfg.model 缺少有效的 action_dim，无法构建流匹配噪声形状。")
    action_keys = tuple(data_config.action_sequence_keys)

    print(colored(f"[infer] LeRobotDataset(repo={data_config.repo_id!r}) ...", "cyan"), flush=True)
    progress.emit("dataset", f"加载 LeRobot 数据集（repo_id={data_config.repo_id}）…")
    dataset = make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = build_repack_only(data_config)
    progress.emit("dataset", f"数据集就绪（共 {len(dataset)} 条）")
    return DatasetBundle(
        train_cfg=train_cfg,
        data_config=data_config,
        dataset=dataset,
        repack_fn=repack_fn,
        action_horizon=int(action_horizon),
        action_dim=int(action_dim),
    )
