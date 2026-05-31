from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class NativeTensorStats:
    num_samples: int
    numel: int
    abs_max: float
    pctl_abs_max: float
    mean_abs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": int(self.num_samples),
            "numel": int(self.numel),
            "abs_max": float(self.abs_max),
            "pctl_abs_max": float(self.pctl_abs_max),
            "mean_abs": float(self.mean_abs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NativeTensorStats":
        return cls(
            num_samples=int(d.get("num_samples", 0)),
            numel=int(d.get("numel", 0)),
            abs_max=float(d.get("abs_max", 0.0)),
            pctl_abs_max=float(d.get("pctl_abs_max", 0.0)),
            mean_abs=float(d.get("mean_abs", 0.0)),
        )


@dataclasses.dataclass
class NativeDecoderQuantSpec:
    version: str
    component: str
    percentile: float
    max_samples: int
    source_calib_data: str
    created_at_utc: str
    tensor_stats: dict[str, NativeTensorStats]
    timestep_stats: dict[str, dict[str, NativeTensorStats]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "component": self.component,
            "percentile": float(self.percentile),
            "max_samples": int(self.max_samples),
            "source_calib_data": self.source_calib_data,
            "created_at_utc": self.created_at_utc,
            "tensor_stats": {k: v.to_dict() for k, v in self.tensor_stats.items()},
            "timestep_stats": {
                tk: {k: v.to_dict() for k, v in tv.items()}
                for tk, tv in self.timestep_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NativeDecoderQuantSpec":
        tensor_stats_raw = d.get("tensor_stats", {}) or {}
        timestep_stats_raw = d.get("timestep_stats", {}) or {}
        tensor_stats = {
            str(k): NativeTensorStats.from_dict(v)
            for k, v in tensor_stats_raw.items()
            if isinstance(v, dict)
        }
        timestep_stats: dict[str, dict[str, NativeTensorStats]] = {}
        for tkey, tv in timestep_stats_raw.items():
            if not isinstance(tv, dict):
                continue
            timestep_stats[str(tkey)] = {
                str(k): NativeTensorStats.from_dict(v)
                for k, v in tv.items()
                if isinstance(v, dict)
            }
        return cls(
            version=str(d.get("version", "native_decoder_quant_spec_v1")),
            component=str(d.get("component", "pi05_denoise")),
            percentile=float(d.get("percentile", 99.9)),
            max_samples=int(d.get("max_samples", 0)),
            source_calib_data=str(d.get("source_calib_data", "")),
            created_at_utc=str(d.get("created_at_utc", "")),
            tensor_stats=tensor_stats,
            timestep_stats=timestep_stats,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NativeDecoderQuantSpec":
        p = Path(path).expanduser().resolve()
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise TypeError(f"Native quant spec root must be object, got {type(raw).__name__}")
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def create(
        cls,
        *,
        component: str,
        percentile: float,
        max_samples: int,
        source_calib_data: str,
        tensor_stats: dict[str, NativeTensorStats],
        timestep_stats: dict[str, dict[str, NativeTensorStats]],
    ) -> "NativeDecoderQuantSpec":
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return cls(
            version="native_decoder_quant_spec_v1",
            component=component,
            percentile=float(percentile),
            max_samples=int(max_samples),
            source_calib_data=source_calib_data,
            created_at_utc=ts,
            tensor_stats=tensor_stats,
            timestep_stats=timestep_stats,
        )

