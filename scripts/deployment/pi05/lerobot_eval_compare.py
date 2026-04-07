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
      --config pi05_libero \\
      --num-samples 50 \\
      --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero

TensorRT 示例::

    python scripts/deployment/pi05/lerobot_eval_compare.py \\
      --checkpoint /path/to/checkpoint \\
      --inference-mode tensorrt \\
      --engine-path /path/to/trt_dir \\
      --vit-engine vit_fp16.trt \\
      --llm-engine llm_fp16.trt \\
      --expert-engine expert_fp16.trt \\
      --denoise-engine denoise.engine

轨迹图（按 LeRobot ``episode_index`` 分 episode，每集一张多子图 PNG）：
**语义与一次推理的 H 步一致**——每隔 ``action_horizon`` 帧做一次 ``infer``，将输出的
``pred[0..H-1]`` 与同一帧 repack 得到的标签 ``gt[0..H-1]`` 逐行对齐，并映射到全局帧
``idx+k``（不重叠）；跨 episode 边界的 chunk 会跳过。红点标在每次推理的起点（chunk 首步）。

::

    python scripts/deployment/pi05/lerobot_eval_compare.py ... \\
      --trajectory-plot-dir ./plots \\
      --max-trajectory-plots 8 \\
      --plot-episodes 0 1
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from termcolor import colored

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


def _unwrap_lerobot_base(dataset) -> object:
    """剥离 ``TransformedDataset`` 等包装，得到底层 ``LeRobotDataset``。"""
    d = dataset
    while hasattr(d, "_dataset"):
        d = d._dataset
    return d


def _global_episode_id_per_frame(base_ds, n_frames: int) -> np.ndarray:
    """长度 ``n_frames``：全局下标 ``i`` 对应 ``episode_index``（与 ``dataset[i]`` 对齐）。"""
    try:
        hf = getattr(base_ds, "hf_dataset", None)
    except Exception as exc:  # pragma: no cover
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
        "未在 hf_dataset 中找到 episode_index；轨迹图将把 [start_index, end) 整段当作 episode_id=0。"
    )
    return np.zeros(n_frames, dtype=np.int64)


def _first_row_state(state, action_dim: int) -> np.ndarray | None:
    """若存在 ``observation/state`` 且最后一维与 action 维一致则返回首步向量，否则 None。"""
    if state is None:
        return None
    s = np.asarray(state, dtype=np.float64)
    if s.ndim == 1:
        row = s
    else:
        row = s[0]
    if row.shape[-1] != action_dim:
        return None
    return row


def _plot_trajectory_episode(
    *,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    state_joints_across_time: np.ndarray | None,
    episode_id: int,
    action_horizon: int,
    save_path: Path,
    index_range: tuple[int, int],
    inference_mode: str,
    infer_start_mask: np.ndarray | None = None,
) -> None:
    """每条轨迹一张 figure：每动作维度一子图（对齐 Isaac-GR00T ``plot_trajectory_results``）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("绘制轨迹需要 matplotlib：pip install matplotlib") from e

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    t = gt_action_across_time.shape[0]
    action_dim = gt_action_across_time.shape[1]
    if t == 0 or action_dim == 0:
        logging.warning("episode %s 无有效帧，跳过作图", episode_id)
        return

    indices = list(range(action_dim))
    fig, axes = plt.subplots(nrows=len(indices), ncols=1, figsize=(8, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    lo, hi = index_range
    fig.suptitle(
        f"Episode {episode_id} | global index [{lo}, {hi}] | backend={inference_mode}",
        fontsize=14,
        color="blue",
    )

    for plot_idx, d in enumerate(indices):
        ax = axes[plot_idx]
        if (
            state_joints_across_time is not None
            and state_joints_across_time.shape == gt_action_across_time.shape
        ):
            ax.plot(state_joints_across_time[:, d], label="state")
        ax.plot(gt_action_across_time[:, d], label="gt action")
        ax.plot(pred_action_across_time[:, d], label="pred action")

        if infer_start_mask is not None and infer_start_mask.shape[0] == t:
            starts = np.flatnonzero(infer_start_mask)
            for si, j in enumerate(starts):
                ax.plot(
                    j,
                    gt_action_across_time[j, d],
                    "ro",
                    label="inference step" if si == 0 else "_nolegend_",
                )
        else:
            for j in range(0, t, max(action_horizon, 1)):
                if j == 0:
                    ax.plot(
                        j,
                        gt_action_across_time[j, d],
                        "ro",
                        label="inference step",
                    )
                else:
                    ax.plot(j, gt_action_across_time[j, d], "ro")

        ax.set_title(f"dim {d}")
        ax.set_xlabel("frame (within episode segment)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@dataclasses.dataclass
class Args:
    """LeRobot 上 pi0.5 预测 vs 标签误差评估。"""

    checkpoint: Path
    """含 ``model.safetensors`` 与 ``assets/`` 的检查点目录。"""

    config: str = "pi05_libero"
    """``openpi.training.config`` 中的 TrainConfig 名称（CLI: ``--config``）。"""

    num_samples: int = 100
    """评估的帧数（数据集下标 0..num_samples-1）。"""

    start_index: int = 0
    """起始全局帧下标。"""

    dataset_root: Path | None = None
    """本地 LeRobot 数据集根目录；为 None 时使用 hub 默认缓存路径。"""

    device: str | None = None
    """PyTorch 设备；默认自动选 CUDA（CLI: ``--device``）。"""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """``pytorch``：原生 PI0Pytorch；``tensorrt``：加载 TRT 引擎并原地替换子模块（见 ``Pi05TensorRTExecutor``）。"""

    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    """TensorRT / 执行器精度；仅 ``inference_mode=tensorrt`` 时使用。"""

    engine_path: str = ""
    """TRT 引擎所在**目录**（CLI: ``--engine-path``；与 ``standalone_inference_script`` 的 ``trt_engine_path`` 一致）。"""

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

    trajectory_plot_dir: Path | None = None
    """若设置，按 **episode_index** 分轨迹各保存一张 PNG（每动作维一子图）；目录会自动创建。
    轨迹按 ``action_horizon`` 步进推理，每行 pred 与同次 infer 的 label 行对齐并铺到连续时间轴。"""

    max_trajectory_plots: int = 32
    """最多生成多少张轨迹图；``0`` 表示不限制。仅 ``trajectory_plot_dir`` 非空时生效。"""

    plot_episodes: tuple[int, ...] = ()
    """仅绘制这些 ``episode_index``；为空则绘制评估范围内出现的所有 episode。"""


def main(args: Args) -> None:
    """入口由 ``tyro.cli(Args)`` 调用，使 CLI 为顶层 ``--checkpoint`` 等，而非 ``--args.*``。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    (
        checkpoint,
        config,
        num_samples,
        start_index,
        dataset_root,
        device,
        inference_mode,
        precision,
        engine_path,
        vit_engine,
        llm_engine,
        expert_engine,
        denoise_engine,
        embed_prefix_engine,
        seed,
        trajectory_plot_dir,
        max_trajectory_plots,
        plot_episodes,
    ) = (
        args.checkpoint,
        args.config,
        args.num_samples,
        args.start_index,
        args.dataset_root,
        args.device,
        args.inference_mode,
        args.precision,
        args.engine_path,
        args.vit_engine,
        args.llm_engine,
        args.expert_engine,
        args.denoise_engine,
        args.embed_prefix_engine,
        args.seed,
        args.trajectory_plot_dir,
        args.max_trajectory_plots,
        args.plot_episodes,
    )

    np.random.seed(seed)

    train_cfg = _config.get_config(config)
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
        dataset_root=dataset_root,
    )
    repack_fn = _build_repack_only(data_config)

    policy = policy_config.create_trained_policy(
        train_cfg,
        checkpoint,
        pytorch_device=device,
    )

    if inference_mode == "tensorrt":
        if not engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
        logging.info(
            "TensorRT：加载引擎目录 %s（vit=%r llm=%r expert=%r denoise=%r embed_prefix=%r）",
            engine_path,
            vit_engine or "(none)",
            llm_engine or "(none)",
            expert_engine or "(none)",
            denoise_engine or "(none)",
            embed_prefix_engine or "(none)",
        )
        _load_tensorrt_engines(
            policy,
            engine_path=engine_path,
            precision=precision,
            vit_engine=vit_engine,
            llm_engine=llm_engine,
            expert_engine=expert_engine,
            denoise_engine=denoise_engine,
            embed_prefix_engine=embed_prefix_engine,
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
    end = min(start_index + num_samples, n)
    if start_index >= n:
        raise ValueError(f"start_index={start_index} >= dataset len={n}")

    base_ds = _unwrap_lerobot_base(dataset)
    ep_per_frame = _global_episode_id_per_frame(base_ds, n)

    mse_list: list[float] = []
    mae_list: list[float] = []
    per_dim_mae: list[np.ndarray] = []

    # 轨迹：按 stride=action_horizon 推理，每步 pred[k] 与 gt[k] 对齐到全局帧 idx+k。
    traj_points: dict[int, list[tuple[int, np.ndarray, np.ndarray, bool, np.ndarray | None]]] = (
        defaultdict(list)
    )
    traj_idx_lo: dict[int, int] = {}
    traj_idx_hi: dict[int, int] = {}

    episode_allow: set[int] | None = set(plot_episodes) if len(plot_episodes) > 0 else None

    for idx in range(start_index, end):
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

        if trajectory_plot_dir is not None:
            stride_ok = (idx - start_index) % action_horizon == 0
            chunk_fits = idx + action_horizon <= n and idx + action_horizon <= end
            if stride_ok and chunk_fits:
                ep0 = int(ep_per_frame[idx])
                ep_last = int(ep_per_frame[idx + action_horizon - 1])
                same_ep = ep0 == ep_last
                if same_ep and (episode_allow is None or ep0 in episode_allow):
                    gt_h = gt_a[:action_horizon]
                    pred_h = pred_a[:action_horizon]
                    if pred_h.shape[0] < action_horizon or gt_h.shape[0] < action_horizon:
                        logging.warning(
                            "index %s: pred/gt 时间维 %s/%s 小于 action_horizon=%s，跳过该段轨迹。",
                            idx,
                            pred_h.shape[0],
                            gt_h.shape[0],
                            action_horizon,
                        )
                    else:
                        st0 = _first_row_state(
                            packed.get("observation/state"),
                            int(gt_h.shape[-1]),
                        )
                        ep_id = ep0
                        for k in range(action_horizon):
                            g = idx + k
                            traj_points[ep_id].append(
                                (
                                    g,
                                    np.asarray(gt_h[k], dtype=np.float64),
                                    np.asarray(pred_h[k], dtype=np.float64),
                                    k == 0,
                                    st0,
                                )
                            )
                            traj_idx_lo[ep_id] = (
                                g if ep_id not in traj_idx_lo else min(traj_idx_lo[ep_id], g)
                            )
                            traj_idx_hi[ep_id] = (
                                g if ep_id not in traj_idx_hi else max(traj_idx_hi[ep_id], g)
                            )

    mse_mean = float(np.mean(mse_list))
    mae_mean = float(np.mean(mae_list))
    mae_std = float(np.std(mae_list))
    per_dim = np.mean(np.stack(per_dim_mae, axis=0), axis=0)

    print(colored(f"推理模式: {inference_mode}", "green"))
    print(
        colored(
            f"样本数: {len(mse_list)} (index {start_index}..{end - 1})",
            "green",
        )
    )
    print(colored(f"MSE(mean over samples): {mse_mean:.6f}", "green"))
    print(colored(f"MAE(mean over samples): {mae_mean:.6f} ± {mae_std:.6f}", "green"))
    print(
        colored(
            "MAE per action dim (mean over samples): "
            + np.array2string(per_dim, precision=4),
            "green",
        )
    )

    if trajectory_plot_dir is not None:
        if not traj_points:
            logging.warning(
                "已设置 trajectory_plot_dir 但未累积轨迹点（检查 --plot-episodes 是否过窄、"
                "评估区间是否不足一整段 action_horizon，或 chunk 是否跨 episode）。"
            )
        ep_sorted = sorted(traj_points.keys(), key=lambda e: (traj_idx_lo[e], e))
        n_plotted = 0
        lim = max_trajectory_plots if max_trajectory_plots > 0 else 10**9
        for ep_id in ep_sorted:
            if n_plotted >= lim:
                logging.info(
                    "已达 max_trajectory_plots=%s，停止保存更多轨迹图。",
                    max_trajectory_plots,
                )
                break
            pts = sorted(traj_points[ep_id], key=lambda p: p[0])
            gts = np.stack([p[1] for p in pts], axis=0)
            prs = np.stack([p[2] for p in pts], axis=0)
            infer_mask = np.array([p[3] for p in pts], dtype=bool)
            st_rows = [p[4] for p in pts]
            state_arr: np.ndarray | None
            if st_rows and all(r is not None for r in st_rows):
                state_arr = np.stack(st_rows, axis=0)  # type: ignore[arg-type]
                if state_arr.shape != gts.shape:
                    state_arr = None
            else:
                state_arr = None
            out_png = Path(trajectory_plot_dir) / (
                f"trajectory_ep{ep_id}_{traj_idx_lo[ep_id]}_{traj_idx_hi[ep_id]}.png"
            )
            _plot_trajectory_episode(
                gt_action_across_time=gts,
                pred_action_across_time=prs,
                state_joints_across_time=state_arr,
                episode_id=ep_id,
                action_horizon=action_horizon,
                save_path=out_png,
                index_range=(traj_idx_lo[ep_id], traj_idx_hi[ep_id]),
                inference_mode=inference_mode,
                infer_start_mask=infer_mask,
            )
            print(colored(f"轨迹图已保存: {out_png}", "green"))
            n_plotted += 1


if __name__ == "__main__":
    # 必须 ``tyro.cli(Args)`` 再传入 main：若写 ``tyro.cli(main)`` 且 ``def main(cli: Args)``，
    # 参数会落在前缀 ``--cli.*`` 下。解析为顶层 ``--checkpoint`` / ``--engine-path`` 等见：
    # https://brentyi.github.io/tyro/
    main(tyro.cli(Args))
