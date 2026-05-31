from __future__ import annotations

from pathlib import Path

from .calibrator import calibrate_native_decoder
from .spec import NativeDecoderQuantSpec


def export_native_decoder_quant_spec(
    *,
    calib_data: str,
    export_path: str,
    component: str = "pi05_denoise",
    percentile: float = 99.9,
    max_samples: int = 0,
) -> NativeDecoderQuantSpec:
    spec = calibrate_native_decoder(
        calib_data=calib_data,
        component=component,
        percentile=percentile,
        max_samples=max_samples,
    )
    p = Path(export_path).expanduser().resolve()
    spec.save(p)
    return spec

