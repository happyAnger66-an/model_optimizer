from __future__ import annotations

import logging
from pathlib import Path

from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize

from .collectors import NativeDecoderStatsCollector
from .spec import NativeDecoderQuantSpec

logger = logging.getLogger(__name__)


def calibrate_native_decoder(
    *,
    calib_data: str,
    component: str = "pi05_denoise",
    percentile: float = 99.9,
    max_samples: int = 0,
) -> NativeDecoderQuantSpec:
    """基于校准数据生成 native decoder 统计规格。"""
    ds = open_pi05_calib_for_quantize(calib_data, component=component)
    collector = NativeDecoderStatsCollector(percentile=percentile)

    num = 0
    for sample in ds:
        collector.update(sample)
        num += 1
        if max_samples > 0 and num >= max_samples:
            break

    g, pt = collector.finalize()
    logger.info(
        "[native-calib] component=%s samples=%s tensors=%s timesteps=%s pctl=%.2f",
        component,
        num,
        len(g),
        len(pt),
        percentile,
    )
    return NativeDecoderQuantSpec.create(
        component=component,
        percentile=percentile,
        max_samples=max_samples,
        source_calib_data=str(Path(calib_data).expanduser().resolve()),
        tensor_stats=g,
        timestep_stats=pt,
    )

