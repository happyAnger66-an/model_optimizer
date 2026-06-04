"""单 chunk 推理共享上下文（从 bundle 解包）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Args
from .running_stats import (
    RunningErrorStats,
    RunningPerDimMsePctStats,
    RunningPerDimPairMseStats,
    RunningPerDimRelP99Stats,
)


@dataclass
class SecondaryPathStats:
    per_dim_mse_pct: RunningPerDimMsePctStats
    per_dim_rel_p99: RunningPerDimRelP99Stats
    pair_mse: RunningPerDimPairMseStats | None = None


@dataclass
class InferChunkContext:
    args: Args
    run_id: str
    dataset: Any
    repack_fn: Any
    policy: Any
    policy_trt: Any | None
    policy_ptq: Any | None
    action_horizon: int
    n: int
    end: int
    start_index: int
    ep_per_frame: np.ndarray
    running_err: RunningErrorStats
    per_dim_mse_pct: RunningPerDimMsePctStats
    per_dim_rel_p99: RunningPerDimRelP99Stats
    trt: SecondaryPathStats | None = None
    ptq: SecondaryPathStats | None = None
    running_vit: Any | None = None
    bundle: dict[str, Any] | None = None

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> InferChunkContext:
        trt = None
        if bundle.get("running_per_dim_mse_pct_trt") is not None:
            trt = SecondaryPathStats(
                per_dim_mse_pct=bundle["running_per_dim_mse_pct_trt"],
                per_dim_rel_p99=bundle["running_per_dim_rel_p99_trt"],
                pair_mse=bundle.get("running_pt_trt_mse_per_dim"),
            )
        ptq = None
        if bundle.get("running_per_dim_mse_pct_ptq") is not None:
            ptq = SecondaryPathStats(
                per_dim_mse_pct=bundle["running_per_dim_mse_pct_ptq"],
                per_dim_rel_p99=bundle["running_per_dim_rel_p99_ptq"],
                pair_mse=bundle.get("running_pt_ptq_mse_per_dim"),
            )
        return cls(
            args=bundle["args"],
            run_id=bundle["run_id"],
            dataset=bundle["dataset"],
            repack_fn=bundle["repack_fn"],
            policy=bundle["policy"],
            policy_trt=bundle.get("policy_trt"),
            policy_ptq=bundle.get("policy_ptq"),
            action_horizon=int(bundle["action_horizon"]),
            n=int(bundle["n"]),
            end=int(bundle["end"]),
            start_index=int(bundle["start_index"]),
            ep_per_frame=bundle["ep_per_frame"],
            running_err=bundle["running_err_stats"],
            per_dim_mse_pct=bundle["running_per_dim_mse_pct"],
            per_dim_rel_p99=bundle["running_per_dim_rel_p99"],
            trt=trt,
            ptq=ptq,
            running_vit=bundle.get("running_vit_pt_trt"),
            bundle=bundle,
        )
