# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""Host 侧 AdaRMS Dense 预计算（roadmap #22 / docs/optimizer/ddup/adarms_pre_compute.md）。

当 denoise 引擎以 AdaRMS 预计算模式导出（输入为 ``adarms_mod`` 而非 ``timestep``）时，
推理宿主需在每步把"打包 modulation"喂给引擎。由于扩散步调度固定，modulation 仅依赖
``timestep``，可按时间步值做记忆化：首次计算、其后命中缓存，dense GEMM 只在 torch 侧
（全精度）跑有限次，彻底移出引擎热路径。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AdaRmsModulator:
    """从 PI0Pytorch 模型预算 / 缓存 denoise 各步的打包 AdaRMS modulation。

    用法::

        mod_fn = AdaRmsModulator(pi05_model)
        adarms_mod = mod_fn(timestep)   # [num_norms, batch, dim*3]，fp32
        engine(..., adarms_mod=adarms_mod)
    """

    def __init__(self, pi05_model: Any) -> None:
        # 以"非预计算"模式构建：dense 权重保持原样、forward 未被替换，仅借其
        # precompute_adarms_modulation 访问真实 time_mlp/dense（全精度）。
        from model_optimizer.models.pi05.dit import Pi05DenoiseStep

        self._step = Pi05DenoiseStep.construct_model(
            pi05_model, adarms_precompute=False
        )
        self._cache: dict[tuple[float, ...], torch.Tensor] = {}

    @staticmethod
    def _key(timestep: torch.Tensor) -> tuple[float, ...]:
        return tuple(round(float(v), 6) for v in timestep.detach().flatten().tolist())

    @torch.no_grad()
    def __call__(self, timestep: torch.Tensor) -> torch.Tensor:
        key = self._key(timestep)
        mod = self._cache.get(key)
        if mod is None:
            mod = self._step.precompute_adarms_modulation(timestep).to(torch.float32)
            self._cache[key] = mod
            logger.debug("AdaRmsModulator: precomputed modulation for t=%s (cache=%d)", key, len(self._cache))
        return mod


def adarms_precompute_enabled(config: Any) -> bool:
    """从 executor 配置 / 环境变量判定是否启用 host 侧 AdaRMS 预计算。

    优先级：``config.denoise_adarms_precompute`` > 环境变量 ``PI05_ADARMS_PRECOMPUTE``。
    """
    import os

    if bool(getattr(config, "denoise_adarms_precompute", False)):
        return True
    return os.environ.get("PI05_ADARMS_PRECOMPUTE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
