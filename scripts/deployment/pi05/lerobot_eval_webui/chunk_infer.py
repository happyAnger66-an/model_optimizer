"""单段 chunk：取样、推理、编码图像、生成 step 事件 JSON。"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
from termcolor import colored

from .dataset import tree_to_numpy
from .infer_backends import select_infer_backend
from .media import encode_jpeg_b64, to_hwc_uint8
from .protocol import StepEvent, event_to_json
from .running_stats import (
    RunningErrorStats,
    RunningPerDimMsePctStats,
    RunningPerDimPairMseStats,
    RunningPerDimRelP99Stats,
)

# 与 standalone / perf 一致：复用同一底层 model 引用打印 time_results
_perf_model: Any | None = None


def _policy_torch_model(policy: Any) -> Any:
    """解析 PyTorch 策略上的 PI0 模块：openpi ``Policy`` 为 ``_model``，少数封装为 ``_policy._model``。"""
    m = getattr(policy, "_model", None)
    if m is not None:
        return m
    inner = getattr(policy, "_policy", None)
    if inner is not None:
        return getattr(inner, "_model", None)
    return None


def flow_match_noise_for_chunk(bundle: dict[str, Any], dataset_chunk_idx: int) -> np.ndarray | None:
    """若 ``args.noise == "fixed"``，返回 ``(action_horizon, action_dim)`` 的 float32 高斯初值；否则 ``None``。

    形状来自 ``bundle``（与 ``train_cfg.model`` 一致），不依赖 ``policy._model``，以便 **纯 TensorRT / ONNXRT** 等路径也能用 ``--noise fixed``。
    """
    args = bundle["args"]
    if getattr(args, "noise", "random") != "fixed":
        return None
    h = int(bundle["action_horizon"])
    d = int(bundle.get("action_dim", 0))
    if d <= 0:
        policy = bundle["policy"]
        model = _policy_torch_model(policy)
        if model is None or not hasattr(model, "config"):
            raise RuntimeError(
                "noise=fixed 需要 bundle['action_dim']（推荐）或可解析的 policy._model.config.action_dim。"
            )
        cfg = model.config
        h = int(cfg.action_horizon)
        d = int(cfg.action_dim)
    seed = int(args.noise_seed)
    ss = np.random.SeedSequence([seed, int(dataset_chunk_idx)])
    rng = np.random.default_rng(ss)
    return rng.standard_normal((h, d), dtype=np.float32)


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
    policy_trt: Any | None = bundle.get("policy_trt")
    policy_ptq: Any | None = bundle.get("policy_ptq")
    running_stats: RunningErrorStats = bundle["running_err_stats"]
    per_dim_mse_pct: RunningPerDimMsePctStats = bundle["running_per_dim_mse_pct"]
    per_dim_rel_p99: RunningPerDimRelP99Stats = bundle["running_per_dim_rel_p99"]
    per_dim_mse_pct_trt: RunningPerDimMsePctStats | None = bundle.get("running_per_dim_mse_pct_trt")
    per_dim_rel_p99_trt: RunningPerDimRelP99Stats | None = bundle.get("running_per_dim_rel_p99_trt")
    pair_mse_per_dim: RunningPerDimPairMseStats | None = bundle.get("running_pt_trt_mse_per_dim")
    per_dim_mse_pct_ptq: RunningPerDimMsePctStats | None = bundle.get("running_per_dim_mse_pct_ptq")
    per_dim_rel_p99_ptq: RunningPerDimRelP99Stats | None = bundle.get("running_per_dim_rel_p99_ptq")
    pair_mse_pt_ptq: RunningPerDimPairMseStats | None = bundle.get("running_pt_ptq_mse_per_dim")
    running_vit = bundle.get("running_vit_pt_trt")

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

    flow_noise = flow_match_noise_for_chunk(bundle, idx)
    backend = select_infer_backend(bundle)
    pack = backend.predict(
        policy, policy_trt, policy_ptq, obs, gt, action_horizon, flow_noise=flow_noise
    )
    infer_ms_pt = pack.infer_ms_pt
    infer_ms_second = pack.infer_ms_second
    pred_h = pack.pred_h
    gt_h = pack.gt_h
    pred_h_trt = pack.pred_h_trt
    pred_h_ptq = pack.pred_h_ptq
    vit_pt_trt = getattr(pack, "vit_pt_trt", None)

    if _perf_model is None:
        _perf_model = _policy_torch_model(policy)

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

    if pred_h.shape[0] < action_horizon or gt_h.shape[0] < action_horizon:
        logging.warning(
            "index %s: pred/gt 时间维 %s/%s 小于 action_horizon=%s，跳过。",
            idx,
            pred_h.shape[0],
            gt_h.shape[0],
            action_horizon,
        )
        return []
    if pred_h_trt is not None and pred_h_trt.shape[0] < action_horizon:
        logging.warning(
            "index %s: pred_trt 时间维 %s 小于 action_horizon=%s，跳过。",
            idx,
            pred_h_trt.shape[0],
            action_horizon,
        )
        return []
    if pred_h_ptq is not None and pred_h_ptq.shape[0] < action_horizon:
        logging.warning(
            "index %s: pred_ptq 时间维 %s 小于 action_horizon=%s，跳过。",
            idx,
            pred_h_ptq.shape[0],
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
        diff_pt = pred_h[k] - gt_h[k]
        dpt_flat = np.ravel(diff_pt.astype(np.float64))
        mse_pt = float(np.mean(diff_pt**2))
        mae_pt = float(np.mean(np.abs(diff_pt)))
        abs_diff = np.abs(diff_pt).astype(np.float64, copy=False)
        abs_gt = np.abs(gt_h[k]).astype(np.float64, copy=False)
        denom = np.maximum(abs_gt, float(args.rel_eps))
        rel_err = (abs_diff / denom).astype(np.float64, copy=False)

        metrics: dict[str, Any] = {
            "mse": mse_pt,
            "mae": mae_pt,
            "mse_pt": mse_pt,
            "mae_pt": mae_pt,
            "mae_per_dim": [float(np.abs(dpt_flat[i])) for i in range(int(dpt_flat.size))],
            "mse_per_dim": [float(dpt_flat[i] * dpt_flat[i]) for i in range(int(dpt_flat.size))],
        }
        # ViT PT vs TRT: attach scalar metrics every step for curve plotting,
        # and update running mean once per chunk (k==0).
        if isinstance(vit_pt_trt, dict) and vit_pt_trt and isinstance(running_vit, object):
            if k == 0 and hasattr(running_vit, "update_from_pack"):
                try:
                    running_vit.update_from_pack(vit_pt_trt)
                except Exception:
                    pass
            cur_mean_abs = vit_pt_trt.get("mean_abs")
            cur_rmse = vit_pt_trt.get("rmse")
            if isinstance(cur_mean_abs, (int, float)):
                metrics["vit_mean_abs"] = float(cur_mean_abs)
            if isinstance(cur_rmse, (int, float)):
                metrics["vit_rmse"] = float(cur_rmse)
            # running (cumulative) mean across chunks
            if hasattr(running_vit, "mean_abs_mean"):
                try:
                    v = running_vit.mean_abs_mean()
                    if v is not None:
                        metrics["vit_mean_abs_cum"] = float(v)
                except Exception:
                    pass
            if hasattr(running_vit, "rmse_mean"):
                try:
                    v = running_vit.rmse_mean()
                    if v is not None:
                        metrics["vit_rmse_cum"] = float(v)
                except Exception:
                    pass
            if k == 0:
                # keep the full pack on chunk start for detailed panel
                metrics["vit_pt_trt"] = vit_pt_trt
        pred_trt_list: list[float] | None = None
        pred_ptq_list: list[float] | None = None
        timing: dict[str, float] | None = None
        if k == 0:
            if infer_ms_second is not None:
                timing = {"infer_ms_pt": float(infer_ms_pt), "infer_ms": float(infer_ms_pt + infer_ms_second)}
                if pred_h_trt is not None:
                    timing["infer_ms_trt"] = float(infer_ms_second)
                if pred_h_ptq is not None:
                    timing["infer_ms_ptq"] = float(infer_ms_second)
            else:
                timing = {"infer_ms": float(infer_ms_pt)}

        if pred_h_trt is not None:
            row_trt = pred_h_trt[k]
            diff_trt = row_trt - gt_h[k]
            diff_pair = pred_h[k] - row_trt
            mse_trt = float(np.mean(diff_trt**2))
            mae_trt = float(np.mean(np.abs(diff_trt)))
            metrics["mse_trt"] = mse_trt
            metrics["mae_trt"] = mae_trt
            metrics["mse_pt_trt"] = float(np.mean(diff_pair**2))
            metrics["mae_pt_trt"] = float(np.mean(np.abs(diff_pair)))
            dpair_flat = np.ravel(diff_pair.astype(np.float64))
            metrics["mae_pt_trt_per_dim"] = [
                float(np.abs(dpair_flat[i])) for i in range(int(dpair_flat.size))
            ]
            metrics["mse_pt_trt_per_dim"] = [
                float(dpair_flat[i] * dpair_flat[i]) for i in range(int(dpair_flat.size))
            ]
            pred_trt_list = [float(x) for x in row_trt.astype(np.float64).tolist()]
            if pair_mse_per_dim is not None:
                pair_mse_per_dim.update(np.ravel(diff_pair).astype(np.float64))
                metrics["mse_pt_trt_dim_mean"] = pair_mse_per_dim.mean_mse_per_dim()

        if pred_h_ptq is not None:
            row_ptq = pred_h_ptq[k]
            diff_ptq = row_ptq - gt_h[k]
            diff_pair_q = pred_h[k] - row_ptq
            mse_ptq_v = float(np.mean(diff_ptq**2))
            mae_ptq_v = float(np.mean(np.abs(diff_ptq)))
            metrics["mse_ptq"] = mse_ptq_v
            metrics["mae_ptq"] = mae_ptq_v
            metrics["mse_pt_ptq"] = float(np.mean(diff_pair_q**2))
            metrics["mae_pt_ptq"] = float(np.mean(np.abs(diff_pair_q)))
            dpair_q = np.ravel(diff_pair_q.astype(np.float64))
            metrics["mae_pt_ptq_per_dim"] = [
                float(np.abs(dpair_q[i])) for i in range(int(dpair_q.size))
            ]
            metrics["mse_pt_ptq_per_dim"] = [
                float(dpair_q[i] * dpair_q[i]) for i in range(int(dpair_q.size))
            ]
            pred_ptq_list = [float(x) for x in row_ptq.astype(np.float64).tolist()]
            if pair_mse_pt_ptq is not None:
                pair_mse_pt_ptq.update(np.ravel(diff_pair_q).astype(np.float64))
                metrics["mse_pt_ptq_dim_mean"] = pair_mse_pt_ptq.mean_mse_per_dim()

        # 全局累计（server 端流式统计，跨 step 累加）— 与单后端一致，仅统计 PyTorch 相对 GT
        running_stats.update_abs_and_rel(
            abs_err_values=np.ravel(abs_diff),
            rel_err_values=np.ravel(rel_err),
        )
        per_dim_mse_pct.update(np.ravel(diff_pt).astype(np.float64), np.ravel(gt_h[k]).astype(np.float64))
        dim_mse_pct_mean = per_dim_mse_pct.mse_pct_mean()
        per_dim_rel_p99.update(np.ravel(rel_err))
        dim_rel_p99 = per_dim_rel_p99.rel_p99()

        metrics["mse_pct_dim_mean"] = dim_mse_pct_mean
        metrics["rel_p99_dim"] = dim_rel_p99

        if (
            pred_h_trt is not None
            and per_dim_mse_pct_trt is not None
            and per_dim_rel_p99_trt is not None
        ):
            abs_diff_trt = np.abs(diff_trt).astype(np.float64, copy=False)
            rel_err_trt = (abs_diff_trt / denom).astype(np.float64, copy=False)
            per_dim_mse_pct_trt.update(
                np.ravel(diff_trt).astype(np.float64), np.ravel(gt_h[k]).astype(np.float64)
            )
            per_dim_rel_p99_trt.update(np.ravel(rel_err_trt))
            metrics["mse_pct_dim_mean_trt"] = per_dim_mse_pct_trt.mse_pct_mean()
            metrics["rel_p99_dim_trt"] = per_dim_rel_p99_trt.rel_p99()

        if (
            pred_h_ptq is not None
            and per_dim_mse_pct_ptq is not None
            and per_dim_rel_p99_ptq is not None
        ):
            abs_diff_ptq = np.abs(diff_ptq).astype(np.float64, copy=False)
            rel_err_ptq = (abs_diff_ptq / denom).astype(np.float64, copy=False)
            per_dim_mse_pct_ptq.update(
                np.ravel(diff_ptq).astype(np.float64), np.ravel(gt_h[k]).astype(np.float64)
            )
            per_dim_rel_p99_ptq.update(np.ravel(rel_err_ptq))
            metrics["mse_pct_dim_mean_ptq"] = per_dim_mse_pct_ptq.mse_pct_mean()
            metrics["rel_p99_dim_ptq"] = per_dim_rel_p99_ptq.rel_p99()

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
            metrics=metrics,
            images=step_images,
            server_timing=timing,
            pred_action_trt=pred_trt_list,
            pred_action_ptq=pred_ptq_list,
        )
        out_msgs.append(event_to_json(dataclasses.asdict(step_event)))
    return out_msgs
