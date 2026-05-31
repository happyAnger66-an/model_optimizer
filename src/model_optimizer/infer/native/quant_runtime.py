from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...quantization.native_decoder.spec import NativeDecoderQuantSpec

logger = logging.getLogger(__name__)


class NativeQuantRuntime:
    """Native decoder 运行时量化规格加载与在线重标定。

    Phase B 第 2 段：
    - 基于 ``NativeQuantSpec`` 读取初始 amax；
    - 对运行时输入执行对称 fake-quant（当前默认作用 ``x_t``）；
    - 可选使用真实数据在线重标定并回写 active amax/scale。
    """

    def __init__(
        self,
        spec: NativeDecoderQuantSpec,
        *,
        recalib_enable: bool = False,
        recalib_max_samples: int = 0,
        recalib_percentile: float = 99.9,
    ) -> None:
        self.spec = spec
        self.recalib_enable = bool(recalib_enable)
        self.recalib_max_samples = max(int(recalib_max_samples), 0)
        self.recalib_percentile = float(recalib_percentile)
        # int8 对称量化，后续可扩展到 fp8/nvfp4 的 scale 语义。
        self._qmax = 127.0
        self._ema = 0.05
        self._apply_tensor_names = {"x_t"}
        self._num_seen = 0
        self._abs_max_seen: dict[str, float] = {}
        self._active_amax: dict[str, float] = {}
        self._active_scale: dict[str, float] = {}
        self._apply_calls = 0
        self._apply_updates = 0
        self._build_initial_state()

    def _build_initial_state(self) -> None:
        for name, st in self.spec.tensor_stats.items():
            # 优先使用 spec 的 percentile 统计，和离线导出保持一致。
            amax = float(max(st.pctl_abs_max, st.abs_max, 0.0))
            if amax <= 0.0:
                continue
            self._active_amax[name] = amax
            self._active_scale[name] = amax / self._qmax
        if self._active_scale:
            logger.info(
                "[native-quant] init active scales from spec: %s",
                ", ".join(
                    f"{k}={v:.6g}" for k, v in sorted(self._active_scale.items())
                ),
            )

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        recalib_enable: bool = False,
        recalib_max_samples: int = 0,
        recalib_percentile: float = 99.9,
    ) -> "NativeQuantRuntime":
        p = Path(path).expanduser().resolve()
        spec = NativeDecoderQuantSpec.load(p)
        logger.info(
            "[native-quant] loaded spec path=%s component=%s tensors=%s timesteps=%s",
            str(p),
            spec.component,
            len(spec.tensor_stats),
            len(spec.timestep_stats),
        )
        return cls(
            spec,
            recalib_enable=recalib_enable,
            recalib_max_samples=recalib_max_samples,
            recalib_percentile=recalib_percentile,
        )

    def observe_sample(self, sample: dict[str, Any]) -> None:
        if not self.recalib_enable:
            return
        if self.recalib_max_samples > 0 and self._num_seen >= self.recalib_max_samples:
            return
        if not isinstance(sample, dict):
            return
        self._num_seen += 1
        for k, v in sample.items():
            if not torch.is_tensor(v) or not v.is_floating_point() or v.numel() == 0:
                continue
            cur = float(v.detach().float().abs().amax().item())
            prev = self._abs_max_seen.get(k, 0.0)
            self._abs_max_seen[k] = max(prev, cur)
            if k in self._active_amax:
                old_amax = float(self._active_amax[k])
                new_amax = (1.0 - self._ema) * old_amax + self._ema * cur
                new_amax = max(new_amax, 1e-8)
                self._active_amax[k] = new_amax
                self._active_scale[k] = new_amax / self._qmax
                self._apply_updates += 1

    def apply_quantized_inputs(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对 denoise 输入应用 runtime scale（当前默认只处理 ``x_t``）。"""
        self._apply_calls += 1
        q_x_t = x_t
        if "x_t" in self._apply_tensor_names:
            scale = float(self._active_scale.get("x_t", 0.0))
            if scale > 0.0:
                q_x_t = torch.clamp(torch.round(x_t / scale), -self._qmax, self._qmax) * scale
                q_x_t = q_x_t.to(dtype=x_t.dtype)
        return prefix_pad_masks, q_x_t, timestep

    def dump_summary(self) -> str:
        lines = ["NativeQuantRuntime summary:"]
        lines.append(
            "  spec: "
            f"component={self.spec.component} tensors={len(self.spec.tensor_stats)} "
            f"timesteps={len(self.spec.timestep_stats)}"
        )
        lines.append(
            "  recalib: "
            f"enable={self.recalib_enable} seen={self._num_seen} "
            f"max_samples={self.recalib_max_samples} pctl={self.recalib_percentile}"
        )
        lines.append(
            "  apply: "
            f"calls={self._apply_calls} updates={self._apply_updates} "
            f"tensors={','.join(sorted(self._apply_tensor_names))}"
        )
        if self._active_scale:
            lines.append(
                "  active_scales: "
                + ", ".join(f"{k}={v:.6g}" for k, v in sorted(self._active_scale.items()))
            )
        if self._abs_max_seen:
            top = sorted(self._abs_max_seen.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append(
                "  recalib_abs_max_top5: "
                + ", ".join(f"{k}={v:.6g}" for k, v in top)
            )
        return "\n".join(lines)

