"""单段 chunk：取样、推理、编码图像、生成 step 事件 JSON。"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

import numpy as np
from termcolor import colored

from .action_align import align_action_dim
from .dataset import tree_to_numpy
from .media import encode_jpeg_b64, to_hwc_uint8
from .protocol import StepEvent, event_to_json
from .running_stats import RunningErrorStats, RunningPerDimMsePctStats, RunningPerDimRelP99Stats

# 与 standalone / perf 一致：复用同一底层 model 引用打印 time_results
_perf_model: Any | None = None


def process_infer_chunk(bundle: dict[str, Any], idx: int) -> list[str]:
    """在专用推理线程中执行单段 chunk：dataset 取样 + infer + 图像编码；返回 JSON 字符串列表。"""
    global _perf_model
    args = bundle["args"]
    action_horizon = bundle["action_horizon"]
    n = bundle["n"]
    end = bundle["end"]
    start_index = bundle["start_index"]
    ep_per_frame = bundle["ep_per_frame"]
    run_id = bundle["run_id"]
    dataset = bundle["dataset"]
    repack_fn = bundle["repack_fn"]
    policy = bundle["policy"]
    running_stats: RunningErrorStats = bundle["running_err_stats"]
    per_dim_mse_pct: RunningPerDimMsePctStats = bundle["running_per_dim_mse_pct"]
    per_dim_rel_p99: RunningPerDimRelP99Stats = bundle["running_per_dim_rel_p99"]

    stride_ok = (idx - start_index) % action_horizon == 0
    chunk_fits = idx + action_horizon <= n and idx + action_horizon <= end
    if not (stride_ok and chunk_fits):
        return []

    ep0 = int(ep_per_frame[idx])
    ep_last = int(ep_per_frame[idx + action_horizon - 1])
    if ep0 != ep_last:
        return []

    raw = tree_to_numpy(dataset[idx])
    packed = repack_fn(dict(raw))
    if "actions" not in packed:
        raise KeyError("repack 后缺少 actions，请检查数据配置与数据集列名是否一致。")

    gt = np.asarray(packed["actions"])
    obs = {k: v for k, v in packed.items() if k != "actions"}

    t0 = time.monotonic()
    out = policy.infer(obs)
    infer_ms = (time.monotonic() - t0) * 1000.0

    if _perf_model is None:
        if hasattr(policy, "_policy"):
            _perf_model = policy._policy._model
        else:
            _perf_model = policy._model

    tr = getattr(_perf_model, "time_results", None) if _perf_model is not None else None
    if tr:
        for key, label in (
            ("suffix", "suffix"),
            ("action", "action"),
            ("vit", "embed_prefix"),
            ("lang_emb", "lang_emb"),
            ("llm", "llm"),
        ):
            if key in tr and tr[key]:
                print(
                    colored(
                        f"{label} {np.mean(tr[key])*1000:.2f} ± {np.std(tr[key])*1000:.2f} ms (shared)",
                        "green",
                    )
                )

    pred = np.asarray(out["actions"])

    pred_a, gt_a = align_action_dim(pred, gt)
    pred_h = pred_a[:action_horizon]
    gt_h = gt_a[:action_horizon]
    if pred_h.shape[0] < action_horizon or gt_h.shape[0] < action_horizon:
        logging.warning(
            "index %s: pred/gt 时间维 %s/%s 小于 action_horizon=%s，跳过。",
            idx,
            pred_h.shape[0],
            gt_h.shape[0],
            action_horizon,
        )
        return []

    prompt: str | None = None
    if "prompt" in packed:
        try:
            prompt = str(packed["prompt"])
        except Exception:
            prompt = None

    images: dict[str, str] | None = None
    if "observation/image" in packed:
        try:
            base_rgb = to_hwc_uint8(packed["observation/image"])
            images = {
                "base_rgb_jpeg_b64": encode_jpeg_b64(base_rgb, quality=args.jpeg_quality),
            }
            if args.send_wrist and "observation/wrist_image" in packed:
                wrist_rgb = to_hwc_uint8(packed["observation/wrist_image"])
                images["wrist_rgb_jpeg_b64"] = encode_jpeg_b64(
                    wrist_rgb, quality=args.jpeg_quality
                )
        except Exception as exc:
            logging.warning("index %s: 图像编码失败（继续只发数值）: %s", idx, exc)
            images = None

    out_msgs: list[str] = []
    for k in range(action_horizon):
        g = idx + k
        diff = pred_h[k] - gt_h[k]
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        abs_diff = np.abs(diff).astype(np.float64, copy=False)
        abs_gt = np.abs(gt_h[k]).astype(np.float64, copy=False)
        denom = np.maximum(abs_gt, float(args.rel_eps))
        rel_err = (abs_diff / denom).astype(np.float64, copy=False)

        # 全局累计（server 端流式统计，跨 step 累加）
        running_stats.update_abs_and_rel(
            abs_err_values=np.ravel(abs_diff),
            rel_err_values=np.ravel(rel_err),
        )
        per_dim_mse_pct.update(np.ravel(diff).astype(np.float64), np.ravel(gt_h[k]).astype(np.float64))
        dim_mse_pct_mean = per_dim_mse_pct.mse_pct_mean()
        per_dim_rel_p99.update(np.ravel(rel_err))
        dim_rel_p99 = per_dim_rel_p99.rel_p99()

        step_images = images if k == 0 else None
        step_event = StepEvent(
            type="step",
            run_id=run_id,
            episode_id=ep0,
            global_index=int(g),
            k_in_chunk=int(k),
            is_chunk_start=bool(k == 0),
            action_horizon=int(action_horizon),
            prompt=prompt if k == 0 else None,
            gt_action=[float(x) for x in gt_h[k].astype(np.float64).tolist()],
            pred_action=[float(x) for x in pred_h[k].astype(np.float64).tolist()],
            metrics={
                "mse": mse,
                "mae": mae,
                # 量化口径：每个动作维度的累计 MSE 百分比（mean(diff^2)/mean(gt^2)）
                "mse_pct_dim_mean": dim_mse_pct_mean,
                # 相对误差：每个动作维度累计 p99（最差 1%）
                "rel_p99_dim": dim_rel_p99,
            },
            images=step_images,
            server_timing={"infer_ms": float(infer_ms)} if k == 0 else None,
        )
        out_msgs.append(event_to_json(dataclasses.asdict(step_event)))
    return out_msgs
