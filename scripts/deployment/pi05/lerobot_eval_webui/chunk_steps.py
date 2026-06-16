"""chunk 内逐步 StepEvent JSON 生成。"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from .chunk_context import InferChunkContext
from .chunk_metrics import StepMetricsAssembler
from .chunk_sample import ChunkSample
from .infer_backends import PredictionPack
from .protocol import StepEvent, event_to_json


def build_step_timing(pack: PredictionPack, k: int) -> dict[str, float] | None:
    if k != 0:
        return None
    if pack.infer_ms_second is not None:
        timing: dict[str, float] = {
            "infer_ms_pt": float(pack.infer_ms_pt),
            "infer_ms": float(pack.infer_ms_pt + pack.infer_ms_second),
        }
        if pack.pred_h_trt is not None:
            timing["infer_ms_trt"] = float(pack.infer_ms_second)
        if pack.pred_h_ptq is not None:
            timing["infer_ms_ptq"] = float(pack.infer_ms_second)
        return timing
    return {"infer_ms": float(pack.infer_ms_pt)}


def emit_chunk_steps(
    ctx: InferChunkContext,
    sample: ChunkSample,
    pack: PredictionPack,
    *,
    idx: int,
    ep0: int,
    images: dict[str, str] | None,
) -> list[str]:
    vit_pt_trt = getattr(pack, "vit_pt_trt", None)
    assembler = StepMetricsAssembler(ctx, pack, vit_pt_trt=vit_pt_trt)
    ah = ctx.action_horizon
    out_msgs: list[str] = []

    for k in range(ah):
        g = idx + k
        pred_row = pack.pred_h[k]
        gt_row = pack.gt_h[k]
        if np.isnan(np.asarray(pred_row, dtype=np.float64)).any():
            logging.warning(
                "chunk idx=%s k=%s global_index=%s: pred 含 NaN，WebUI 将显示 pred 为空；"
                "请更新 model_optimizer（stage_perf 已对齐 Observation.from_dict）并重试",
                idx,
                k,
                idx + k,
            )
        result = assembler.build(k, pred_row, gt_row)
        step_images = images if k == 0 else None
        step_event = StepEvent(
            type="step",
            run_id=ctx.run_id,
            episode_id=ep0,
            global_index=int(g),
            k_in_chunk=int(k),
            is_chunk_start=bool(k == 0),
            action_horizon=int(ah),
            prompt=sample.prompt if k == 0 else None,
            gt_action=[float(x) for x in gt_row.astype(np.float64).tolist()],
            pred_action=[float(x) for x in pred_row.astype(np.float64).tolist()],
            metrics=result.metrics,
            images=step_images,
            server_timing=build_step_timing(pack, k),
            pred_action_trt=result.pred_trt_list,
            pred_action_ptq=result.pred_ptq_list,
        )
        out_msgs.append(event_to_json(dataclasses.asdict(step_event)))
    return out_msgs
