#!/usr/bin/env python3
"""
在开源 LeRobot 数据上运行 pi0.5 策略，并将预测动作与数据集中的标签做误差对比。

数据管线与 ``openpi.training.data_loader.create_torch_dataset`` 一致：先按帧取原始样本，
可选 ``PromptFromLeRobotTask``，再应用与训练相同的 ``repack``，得到与 ``LiberoInputs`` 一致的键
（``observation/image`` 等），构造 ``obs`` 调用 ``policy.infer``；标签取 repack 后的 ``actions``
chunk，与 ``infer`` 输出中经 ``LiberoOutputs`` 截断后的 ``actions``（前 7 维）对齐比较。

依赖：需安装 ``lerobot``；``openpi`` 需在 PYTHONPATH 中。使用 TensorRT 时还需能 import
``model_optimizer``（例如 ``pip install -e .`` 或将仓库 ``src`` 加入 PYTHONPATH）。

推理后端 ``--inference-mode``：``pytorch``（默认）或 ``tensorrt``。TensorRT 路径与
``standalone_inference_script.py`` 相同：``Pi05TensorRTExecutor.load_model`` 会**原地**把
``policy._model`` 中对应子模块换成引擎，随后仍调用 ``policy.infer``。

示例::

    PYTHONPATH=third_party/openpi/src:$PYTHONPATH \\
    python scripts/deployment/pi05/lerobot_eval_compare.py \\
      --checkpoint /path/to/checkpoint \\
      --config-name pi05_libero \\
      --num-samples 50 \\
      --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero

TensorRT 示例::

    python scripts/deployment/pi05/lerobot_eval_compare.py \\
      --checkpoint /path/to/checkpoint \\
      --inference-mode tensorrt \\
      --trt-engine-path /path/to/trt_dir \\
      --vit-engine vit_fp16.trt \\
      --llm-engine llm_fp16.trt \\
      --expert-engine expert_fp16.trt \\
      --denoise-engine denoise.engine
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

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


def _load_tensorrt_engines(
    policy,
    *,
    engine_path: str,
    precision: Literal["fp16", "bf16", "fp32"],
    vit_engine: str,
    llm_engine: str,
    expert_engine: str,
    denoise_engine: str,
    embed_prefix_engine: str,
) -> None:
    """与 ``standalone_inference_script`` 一致：原地 patch ``policy._model`` 后仍用 ``policy.infer``。"""
    import addict
    import torch

    from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

    if precision == "fp16":
        prec = torch.float16
    elif precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    executor = Pi05TensorRTExecutor(policy, prec)
    cfg: dict = {"engine_path": engine_path}
    if vit_engine:
        cfg["vit_engine"] = vit_engine
    if expert_engine:
        cfg["expert_engine"] = expert_engine
    if llm_engine:
        cfg["llm_engine"] = llm_engine
    if denoise_engine:
        cfg["denoise_engine"] = denoise_engine
    if embed_prefix_engine:
        cfg["embed_prefix_engine"] = embed_prefix_engine

    executor.load_model(addict.Dict(cfg))


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

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """``pytorch``：原生 PI0Pytorch；``tensorrt``：加载 TRT 引擎并原地替换子模块（见 ``Pi05TensorRTExecutor``）。"""

    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    """TensorRT / 执行器精度；仅 ``inference_mode=tensorrt`` 时使用。"""

    trt_engine_path: str = ""
    """TRT 引擎所在**目录**（与 standalone 一致；其下再拼各引擎文件名）。"""

    vit_engine: str = ""
    """ViT 引擎文件名；留空则不替换 ViT。"""

    llm_engine: str = ""
    """LLM 引擎文件名；留空则不替换 language model。"""

    expert_engine: str = ""
    """Expert 引擎文件名；留空则不替换 expert。"""

    denoise_engine: str = ""
    """Denoise 步引擎文件名（如 ``denoise.engine``）；留空则不替换 denoise_step。"""

    embed_prefix_engine: str = ""
    """embed_prefix 引擎文件名；留空则不替换 embed_prefix。"""

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

    if args.inference_mode == "tensorrt":
        if not args.trt_engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --trt-engine-path（引擎目录）。")
        logging.info(
            "TensorRT：加载引擎目录 %s（vit=%r llm=%r expert=%r denoise=%r embed_prefix=%r）",
            args.trt_engine_path,
            args.vit_engine or "(none)",
            args.llm_engine or "(none)",
            args.expert_engine or "(none)",
            args.denoise_engine or "(none)",
            args.embed_prefix_engine or "(none)",
        )
        _load_tensorrt_engines(
            policy,
            engine_path=args.trt_engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
    else:
        logging.info("推理后端: PyTorch（未加载 TensorRT 引擎）。")

    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

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

    logging.info("推理模式: %s", args.inference_mode)
    logging.info("样本数: %d (index %d..%d)", len(mse_list), args.start_index, end - 1)
    logging.info("MSE(mean over samples): %.6f", mse_mean)
    logging.info("MAE(mean over samples): %.6f ± %.6f", mae_mean, mae_std)
    logging.info("MAE per action dim (mean over samples): %s", np.array2string(per_dim, precision=4))


if __name__ == "__main__":
    tyro.cli(main)
