"""chunk 首帧观测图像 JPEG 编码。"""

from __future__ import annotations

import logging
from typing import Any

from .config import Args
from .media import encode_jpeg_b64, to_hwc_uint8


def encode_chunk_images(packed: dict[str, Any], args: Args, idx: int) -> dict[str, str] | None:
    if "observation/image" not in packed:
        return None
    try:
        base_rgb = to_hwc_uint8(packed["observation/image"])
        images: dict[str, str] = {
            "base_rgb_jpeg_b64": encode_jpeg_b64(base_rgb, quality=args.jpeg_quality),
        }
        if args.send_wrist and "observation/wrist_image" in packed:
            wrist_rgb = to_hwc_uint8(packed["observation/wrist_image"])
            images["wrist_rgb_jpeg_b64"] = encode_jpeg_b64(
                wrist_rgb, quality=args.jpeg_quality
            )
        return images
    except Exception as exc:
        logging.warning("index %s: 图像编码失败（继续只发数值）: %s", idx, exc)
        return None
