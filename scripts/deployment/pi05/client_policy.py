#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 向 ``serve_policy.py`` 暴露的 WebSocket 策略服务发送 ``infer``（及可选 ``score``）请求。
# 数据集与 ``--dataset-root`` 解析对齐 ``lerobot_eval_webui``（同目录 ``lerobot_eval_webui/config.py`` /
# ``dataset.py``），便于与离线 WebUI 评估使用同一 LeRobot 数据源。
#
# 示例：
#
#   python scripts/deployment/pi05/client_policy.py \\
#     --config-name pi05_libero \\
#     --dataset-root /path/to/lerobot_data \\
#     --host 127.0.0.1 --port 8000 \\
#     --start-index 0 --num-samples 500
#
#   # 仅探测 HTTP ``/healthz``（与 ``WebsocketPolicyServer`` 一致）
#   python scripts/deployment/pi05/client_policy.py --health-only --host 127.0.0.1 --port 8000
#
# 依赖：``openpi``、``openpi-client``（``WebsocketClientPolicy`` / ``msgpack_numpy``）、
# ``tyro``、``lerobot``（与 WebUI 相同）、可选 ``tqdm``。

from __future__ import annotations

import dataclasses
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    """连接 ``serve_policy`` WebSocket，并按 LeRobot 数据构造与训练配置一致的观测。"""

    config_name: str = "pi05_libero"
    """与 ``serve_policy.py`` / ``lerobot_eval_webui`` 相同：``get_config`` 名或 ``TrainConfig`` 的 ``.py`` 路径。"""

    dataset_root: Path | None = None
    """LeRobot 根目录；传给 ``LeRobotDataset(..., root=...)``，与 WebUI ``--dataset-root`` 一致。"""

    host: str = "127.0.0.1"
    """WebSocket 主机；勿用 ``0.0.0.0`` 作为客户端目标。"""

    port: int = 8000
    api_key: str | None = None

    start_index: int = 0
    num_samples: int = 500
    """与 WebUI 一致：在 ``[start_index, start_index + num_samples)`` 与 ``len(dataset)`` 交集中取帧。"""

    rel_eps: float = 1e-8
    """相对误差分母 ``max(|gt|, rel_eps)``，与 WebUI ``--rel-eps`` 一致。"""

    score_once: bool = False
    """若 True 且服务端 ``supports_score_endpoint``，对首个 chunk 的观测额外发一条 ``score`` 请求。"""

    health_only: bool = False
    """为 True 时仅请求 ``http://{host}:{port}/healthz`` 后退出（不加载数据集）。"""

    quiet: bool = False


def _http_healthz(host: str, port: int) -> str:
    url = f"http://{host}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            return resp.read().decode("utf-8", errors="replace").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"GET {url!r} failed: {e}") from e


def _chunk_indices(
    *,
    start_index: int,
    end: int,
    n: int,
    action_horizon: int,
    ep_per_frame: np.ndarray,
) -> list[int]:
    """与 ``lerobot_eval_webui.chunk_infer.process_infer_chunk`` 相同的 chunk 起点筛选。"""
    end = min(end, n)
    out: list[int] = []
    for idx in range(start_index, end):
        stride_ok = (idx - start_index) % action_horizon == 0
        chunk_fits = idx + action_horizon <= n and idx + action_horizon <= end
        if not (stride_ok and chunk_fits):
            continue
        ep0 = int(ep_per_frame[idx])
        ep_last = int(ep_per_frame[idx + action_horizon - 1])
        if ep0 != ep_last:
            continue
        out.append(idx)
    return out


def main(args: Args) -> None:
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, force=True)

    if args.health_only:
        body = _http_healthz(args.host, args.port)
        print(f"healthz OK: {body!r}")
        return

    from lerobot_eval_webui.dataset import (
        build_repack_only,
        global_episode_id_per_frame,
        make_lerobot_dataset,
        tree_to_numpy,
        unwrap_lerobot_base,
    )
    from openpi_client import websocket_client_policy as _websocket_client_policy

    import serve_policy as _serve_policy

    train_cfg = _serve_policy.load_train_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前 TrainConfig.data 未设置 repo_id，无法加载 LeRobot 数据集。")

    action_horizon = int(train_cfg.model.action_horizon)
    action_keys = tuple(data_config.action_sequence_keys)

    dataset = make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = build_repack_only(data_config)

    n = len(dataset)
    base_ds = unwrap_lerobot_base(dataset)
    ep_per_frame = global_episode_id_per_frame(base_ds, n)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    indices = _chunk_indices(
        start_index=args.start_index,
        end=end,
        n=n,
        action_horizon=action_horizon,
        ep_per_frame=ep_per_frame,
    )
    if not indices:
        logging.warning(
            "无可用 chunk（检查 start_index、num_samples、action_horizon=%s 与 episode 边界）。",
            action_horizon,
        )

    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port, api_key=args.api_key
    )
    meta = client.get_server_metadata()
    if not args.quiet:
        print("server metadata:", meta)

    supports_score = bool(meta.get("supports_score_endpoint"))
    if args.score_once and not supports_score:
        logging.warning("已设 --score-once 但服务端未声明 supports_score_endpoint，跳过 score。")

    process_ms: list[float] = []
    mae_list: list[float] = []
    mse_list: list[float] = []
    rel_mae_list: list[float] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # type: ignore[assignment, misc]

    for chunk_i, idx in enumerate(tqdm(indices, desc="remote infer")):
        raw = tree_to_numpy(dataset[idx])
        packed = repack_fn(dict(raw))
        if "actions" not in packed:
            raise KeyError("repack 后缺少 actions，请检查数据配置与数据集列名。")

        gt = np.asarray(packed["actions"])
        obs: dict[str, Any] = {k: v for k, v in packed.items() if k != "actions"}

        if args.score_once and supports_score and chunk_i == 0:
            try:
                score_out = client.infer({**obs, "_request_type": "score"})
                if not args.quiet:
                    print(
                        "score (first chunk):",
                        {k: score_out[k] for k in score_out if k != "value_logits"},
                    )
            except Exception as e:
                logging.warning("score 请求失败: %s", e)

        out = client.infer(obs)
        st = out.get("server_timing") or {}
        if "process_ms" in st:
            process_ms.append(float(st["process_ms"]))

        pred = np.asarray(out["actions"])
        if pred.shape != gt.shape:
            logging.warning("idx=%s pred shape %s != gt %s，跳过误差统计。", idx, pred.shape, gt.shape)
            continue

        diff = pred.astype(np.float64) - gt.astype(np.float64)
        mae_list.append(float(np.mean(np.abs(diff))))
        mse_list.append(float(np.mean(diff**2)))
        denom = np.maximum(np.abs(gt.astype(np.float64)), float(args.rel_eps))
        rel_mae_list.append(float(np.mean(np.abs(diff) / denom)))

    if process_ms:
        print(
            "server_timing process_ms: "
            f"mean={np.mean(process_ms):.2f} std={np.std(process_ms):.2f} "
            f"n={len(process_ms)}"
        )
    if mae_list:
        print(
            "vs dataset GT (same chunk): "
            f"mae_mean={np.mean(mae_list):.6f} mse_mean={np.mean(mse_list):.6f} "
            f"rel_mae_mean={np.mean(rel_mae_list):.6f} chunks={len(mae_list)}"
        )


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        sys.exit(130)
