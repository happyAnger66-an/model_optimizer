"""观测图像 JPEG 编码（WebUI 推送）。"""

from __future__ import annotations

import base64

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0.0, 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def encode_jpeg_b64(rgb_hwc_uint8: np.ndarray, *, quality: int = 85) -> str:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("缺少 opencv-python-headless，无法编码 JPEG（requirements-base.txt 已包含）。")
    img = np.asarray(rgb_hwc_uint8)
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expect uint8(H,W,3), got {img.dtype} {img.shape}")
    bgr = img[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
