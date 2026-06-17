"""chunk 推理与预测形状校验。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunk_context import InferChunkContext
from .chunk_policy import policy_torch_model
from .chunk_sample import ChunkSample
from .infer_backends import PredictionPack, select_infer_backend


def flow_match_noise_for_chunk(bundle: dict[str, Any], dataset_chunk_idx: int) -> np.ndarray | None:
    """若 ``args.noise == "fixed"``，返回 ``(action_horizon, action_dim)`` 的 float32 高斯初值；否则 ``None``。"""
    args = bundle["args"]
    if getattr(args, "noise", "random") != "fixed":
        return None
    h = int(bundle["action_horizon"])
    d = int(bundle.get("action_dim", 0))
    if d <= 0:
        policy = bundle["policy"]
        model = policy_torch_model(policy)
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


@dataclass
class ChunkPrediction:
    pack: PredictionPack
    t_predict_0: float
    t_after_predict: float


def _prediction_horizon_ok(
    pred: np.ndarray | None,
    *,
    idx: int,
    action_horizon: int,
    label: str,
) -> bool:
    if pred is None:
        return True
    if pred.shape[0] < action_horizon:
        logging.warning(
            "index %s: pred_%s 时间维 %s 小于 action_horizon=%s，跳过。",
            idx,
            label,
            pred.shape[0],
            action_horizon,
        )
        return False
    return True


def run_chunk_prediction(
    ctx: InferChunkContext,
    sample: ChunkSample,
    idx: int,
) -> ChunkPrediction | None:
    bundle = ctx.bundle
    if bundle is None:
        raise RuntimeError("InferChunkContext.bundle is required for inference")
    flow_noise = flow_match_noise_for_chunk(bundle, idx)
    backend = select_infer_backend(bundle)
    t_predict_0 = time.perf_counter()
    pack = backend.predict(
        ctx.policy,
        ctx.policy_trt,
        ctx.policy_ptq,
        sample.obs,
        sample.gt_h,
        ctx.action_horizon,
        flow_noise=flow_noise,
    )
    t_after_predict = time.perf_counter()
    from model_optimizer.infer.perf.gpu_memory import gpu_mem_report_after_first_infer_once

    gpu_mem_report_after_first_infer_once()
    ah = ctx.action_horizon
    if pack.pred_h.shape[0] < ah or pack.gt_h.shape[0] < ah:
        logging.warning(
            "index %s: pred/gt 时间维 %s/%s 小于 action_horizon=%s，跳过。",
            idx,
            pack.pred_h.shape[0],
            pack.gt_h.shape[0],
            ah,
        )
        return None
    if not _prediction_horizon_ok(pack.pred_h_trt, idx=idx, action_horizon=ah, label="trt"):
        return None
    if not _prediction_horizon_ok(pack.pred_h_ptq, idx=idx, action_horizon=ah, label="ptq"):
        return None
    return ChunkPrediction(pack=pack, t_predict_0=t_predict_0, t_after_predict=t_after_predict)
