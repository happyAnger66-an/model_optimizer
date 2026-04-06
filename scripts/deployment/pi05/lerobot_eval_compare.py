#!/usr/bin/env python3
"""
在开源 LeRobot 数据上运行 pi0.5 策略，并将预测动作与数据集中的标签做误差对比。

数据管线与 ``openpi.training.data_loader.create_torch_dataset`` 一致：先按帧取原始样本，
可选 ``PromptFromLeRobotTask``，再应用与训练相同的 ``repack``，得到与 ``LiberoInputs`` 一致的键
（``observation/image`` 等），构造 ``obs`` 调用 ``policy.infer``；标签取 repack 后的 ``actions``
chunk，与 ``infer`` 输出中经 ``LiberoOutputs`` 截断后的 ``actions``（前 7 维）对齐比较。

依赖：需安装 ``lerobot``，且能将 ``openpi`` 加入 PYTHONPATH（例如
``export PYTHONPATH=third_party/openpi/src:$PYTHONPATH``）。

示例::

    PYTHONPATH=third_party/openpi/src:$PYTHONPATH \\
    python scripts/deployment/pi05/lerobot_eval_compare.py \\
      --checkpoint /path/to/checkpoint \\
      --config-name pi05_libero \\
      --num-samples 50 \\
      --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import tyro

import openpi.transforms as _transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training.data_loader import TransformedDataset

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:  # pragma: no cover
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError(
            "需要安装 lerobot（openpi 使用 lerobot.common.datasets.lerobot_dataset；"
            "新版包可能为 lerobot.datasets.lerobot_dataset）。"
        ) from e


def _build_repack_only(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose([*data_config.repack_transforms.inputs])


def _tree_to_numpy(obj):
    """LeRobot 常返回 torch.Tensor；openpi transforms 期望 numpy。"""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to_numpy(x) for x in obj)
    return obj


def _make_lerobot_dataset(
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


def _align_action_dim(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    d = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :d], gt[..., :d]


@dataclasses.dataclass
class Args:
    """LeRobot 上 pi0.5 预测 vs 标签误差评估。"""

    checkpoint: Path
    """含 ``model.safetensors`` 与 ``assets/`` 的检查点目录。"""

    config_name: str = "pi05_libero"
    """``openpi.training.config`` 中的 TrainConfig 名称。"""

    num_samples: int = 100
    """评估的帧数（数据集下标 0..num_samples-1）。"""

    start_index: int = 0
    """起始全局帧下标。"""

    dataset_root: Path | None = None
    """本地 LeRobot 数据集根目录；为 None 时使用 hub 默认缓存路径。"""

    pytorch_device: str | None = None
    """PyTorch 设备；默认自动选 CUDA。"""

    seed: int = 0
    """NumPy 随机种子（若策略含随机采样）。"""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(args.seed)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_keys = tuple(data_config.action_sequence_keys)

    dataset = _make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = _build_repack_only(data_config)

    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        pytorch_device=args.pytorch_device,
    )

    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    mse_list: list[float] = []
    mae_list: list[float] = []
    per_dim_mae: list[np.ndarray] = []

    for idx in range(args.start_index, end):
        raw = _tree_to_numpy(dataset[idx])
        packed = repack_fn(dict(raw))
        if "actions" not in packed:
            raise KeyError("repack 后缺少 actions，请检查数据配置与数据集列名是否一致。")
        gt = np.asarray(packed["actions"])
        obs = {k: v for k, v in packed.items() if k != "actions"}
        out = policy.infer(obs)
        pred = np.asarray(out["actions"])

        pred_a, gt_a = _align_action_dim(pred, gt)
        diff = pred_a - gt_a
        mse_list.append(float(np.mean(diff**2)))
        mae_list.append(float(np.mean(np.abs(diff))))
        per_dim_mae.append(np.mean(np.abs(diff), axis=tuple(range(diff.ndim - 1))))

    mse_mean = float(np.mean(mse_list))
    mae_mean = float(np.mean(mae_list))
    mae_std = float(np.std(mae_list))
    per_dim = np.mean(np.stack(per_dim_mae, axis=0), axis=0)

    logging.info("样本数: %d (index %d..%d)", len(mse_list), args.start_index, end - 1)
    logging.info("MSE(mean over samples): %.6f", mse_mean)
    logging.info("MAE(mean over samples): %.6f ± %.6f", mae_mean, mae_std)
    logging.info("MAE per action dim (mean over samples): %s", np.array2string(per_dim, precision=4))


if __name__ == "__main__":
    tyro.cli(main)
