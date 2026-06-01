from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_hwc_image(x: Any) -> np.ndarray:
    """统一图像到 HWC numpy，dtype 保持输入语义。"""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"Expected image rank=3 after normalize, got shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _stringify_prompt(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return str(v.item())
        if v.size == 1:
            return str(v.reshape(-1)[0])
    return str(v)


class FlashRtPolicyAdapter:
    """把 FlashRT 前端适配为 ``policy.infer(obs)`` 语义。"""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        num_views: int = 2,
        use_cuda_graph: bool = True,
        autotune: int = 3,
        use_fp8: bool = True,
        recalibrate_with_real_data: bool = False,
    ) -> None:
        from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

        self.frontend = Pi05TorchFrontendThor(
            checkpoint_dir,
            num_views=int(num_views),
            use_cuda_graph=bool(use_cuda_graph),
            autotune=int(autotune),
            use_fp8=bool(use_fp8),
        )
        self._last_prompt: str | None = None
        if not recalibrate_with_real_data:
            # 关闭首帧真实数据重标定：保持离线/显式 calibrate 的可复现性。
            try:
                self.frontend._real_data_calibrated = True  # noqa: SLF001
            except Exception:
                pass

    @staticmethod
    def _extract_prompt(obs: dict[str, Any]) -> str | None:
        for key in ("prompt", "instruction", "task"):
            if key in obs and obs[key] is not None:
                return _stringify_prompt(obs[key])
        return None

    @staticmethod
    def _extract_images(obs: dict[str, Any]) -> dict[str, Any]:
        if "images" in obs and obs["images"] is not None:
            imgs = [_to_hwc_image(im) for im in list(obs["images"])]
            return {"images": imgs}
        img = obs.get("image", obs.get("observation/image"))
        if img is None:
            raise KeyError("FlashRT infer requires image or observation/image")
        out: dict[str, Any] = {"image": _to_hwc_image(img)}
        wrist = obs.get("wrist_image", obs.get("observation/wrist_image"))
        if wrist is not None:
            out["wrist_image"] = _to_hwc_image(wrist)
        return out

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = self._extract_prompt(obs)
        if prompt is not None and prompt != self._last_prompt:
            self.frontend.set_prompt(prompt)
            self._last_prompt = prompt
        flash_obs = self._extract_images(obs)
        return self.frontend.infer(flash_obs)

    def calibrate(self, observations: list[dict[str, Any]], *, percentile: float, max_samples: int) -> None:
        if not observations:
            return
        norm_obs = [self._extract_images(o) for o in observations]
        self.frontend.calibrate(
            norm_obs,
            percentile=float(percentile),
            max_samples=int(max_samples) if max_samples > 0 else None,
            verbose=True,
        )
        logger.info(
            "[flashrt] calibration done: n=%s percentile=%.2f max_samples=%s",
            len(norm_obs),
            percentile,
            max_samples,
        )

