from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from .spec import NativeTensorStats


@dataclasses.dataclass
class _TensorAccumulator:
    num_samples: int = 0
    numel: int = 0
    abs_max: float = 0.0
    sample_abs_max: list[float] = dataclasses.field(default_factory=list)
    sum_abs: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x):
            return
        if not x.is_floating_point():
            return
        if x.numel() == 0:
            return
        xf = x.detach().float()
        cur_abs_max = float(xf.abs().amax().item())
        cur_sum_abs = float(xf.abs().sum().item())
        self.num_samples += 1
        self.numel += int(xf.numel())
        self.abs_max = max(self.abs_max, cur_abs_max)
        self.sample_abs_max.append(cur_abs_max)
        self.sum_abs += cur_sum_abs

    def finalize(self, percentile: float) -> NativeTensorStats:
        if self.num_samples <= 0 or self.numel <= 0:
            return NativeTensorStats(0, 0, 0.0, 0.0, 0.0)
        pctl = float(np.percentile(np.asarray(self.sample_abs_max, dtype=np.float64), percentile))
        mean_abs = float(self.sum_abs / max(self.numel, 1))
        return NativeTensorStats(
            num_samples=int(self.num_samples),
            numel=int(self.numel),
            abs_max=float(self.abs_max),
            pctl_abs_max=float(pctl),
            mean_abs=float(mean_abs),
        )


class NativeDecoderStatsCollector:
    """收集 native decoder 校准统计。

    注：Phase B 起步版使用“每样本 abs_max 的 percentile”，避免存全量激活导致内存爆炸。
    """

    def __init__(self, *, percentile: float = 99.9) -> None:
        self.percentile = float(percentile)
        self._global: dict[str, _TensorAccumulator] = {}
        self._per_timestep: dict[str, dict[str, _TensorAccumulator]] = {}

    @staticmethod
    def _extract_timestep_key(sample: dict[str, Any]) -> str:
        t = sample.get("timestep")
        if torch.is_tensor(t) and t.numel() > 0:
            tv = int(t.detach().float().reshape(-1)[0].item())
            return str(tv)
        return "unknown"

    def update(self, sample: dict[str, Any]) -> None:
        if not isinstance(sample, dict):
            return
        tkey = self._extract_timestep_key(sample)
        if tkey not in self._per_timestep:
            self._per_timestep[tkey] = {}
        per_t = self._per_timestep[tkey]
        for k, v in sample.items():
            if not torch.is_tensor(v) or not v.is_floating_point():
                continue
            if k not in self._global:
                self._global[k] = _TensorAccumulator()
            if k not in per_t:
                per_t[k] = _TensorAccumulator()
            self._global[k].update(v)
            per_t[k].update(v)

    def finalize(self) -> tuple[dict[str, NativeTensorStats], dict[str, dict[str, NativeTensorStats]]]:
        g = {k: acc.finalize(self.percentile) for k, acc in self._global.items()}
        pt: dict[str, dict[str, NativeTensorStats]] = {}
        for tk, m in self._per_timestep.items():
            pt[tk] = {k: acc.finalize(self.percentile) for k, acc in m.items()}
        return g, pt

