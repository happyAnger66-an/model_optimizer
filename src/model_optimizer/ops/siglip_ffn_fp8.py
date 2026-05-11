# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP FFN FP8 融合路径的 PyTorch 参考实现（与 ``SiglipFfFp8Plugin`` / FlashRT 数学顺序对齐）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def siglip_ffn_fp8_eager(
    x_fp8: torch.Tensor,
    residual: torch.Tensor,
    up_w_fp8: torch.Tensor,
    down_w_fp8: torch.Tensor,
    up_b: torch.Tensor,
    down_b: torch.Tensor,
    unit_scale: torch.Tensor,
    alpha_up: float = 1.0,
    alpha_down: float = 1.0,
) -> torch.Tensor:
    """参考前向：Up GEMM（FP8 权重按元素转 float）→ GELU →静态 FP8 量化 → Down GEMM + 残差。

    与 C++ 插件中 ``mopt_siglip_ffn_enqueue`` 顺序一致；数值与 cuBLASLt FP8 核不完全逐 bit 一致，
    但用于端到端测试的相对误差通常足够小（见测试中的 ``rtol`` / ``atol``）。
    """

    if x_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"x_fp8 must be torch.float8_e4m3fn, got {x_fp8.dtype}")
    if up_w_fp8.dtype != torch.float8_e4m3fn or down_w_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("up_w_fp8 / down_w_fp8 must be torch.float8_e4m3fn")
    x = x_fp8.to(torch.float32)
    uw = up_w_fp8.to(torch.float32)
    dw = down_w_fp8.to(torch.float32)
    ub = up_b.to(torch.float32)
    db = down_b.to(torch.float32)
    res = residual.to(torch.float32)
    hidden = F.gelu(float(alpha_up) * (x @ uw) + ub)
    scale = unit_scale.to(hidden.device, dtype=torch.float32).reshape(-1)[0]
    inv = 1.0 / torch.clamp(scale, min=1e-12)
    hid_q = (hidden * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    y = res + float(alpha_down) * (hid_q.to(torch.float32) @ dw) + db
    return y.to(residual.dtype)


class TinySiglipFfFp8Mlp(nn.Module):
    """最小可测模型：仅包含一路 FP8 FFN（Up/Down 权重为 FP8 buffer）。"""

    def __init__(self, d: int, h: int, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.d = int(d)
        self.h = int(h)
        dev = device or torch.device("cpu")
        wu = torch.randn(d, h, device=dev, dtype=torch.float32)
        wd = torch.randn(h, d, device=dev, dtype=torch.float32)
        self.register_buffer("up_w", wu.to(torch.float8_e4m3fn))
        self.register_buffer("down_w", wd.to(torch.float8_e4m3fn))
        self.up_b = nn.Parameter(torch.randn(h, device=dev, dtype=torch.float16))
        self.down_b = nn.Parameter(torch.randn(d, device=dev, dtype=torch.float16))
        self.register_buffer("unit_scale", torch.tensor([2.0], device=dev, dtype=torch.float32))

    def forward(self, x_fp8: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return siglip_ffn_fp8_eager(
            x_fp8,
            residual,
            self.up_w,
            self.down_w,
            self.up_b,
            self.down_b,
            self.unit_scale,
            1.0,
            1.0,
        )


def discover_siglip_ffn_fp8_plugin_so(extra_paths: Iterable[str | Path] | None = None) -> str | None:
    """解析 ``libtrt_siglip_ffn_fp8_plugin.so`` 路径。

    优先级：

    1. 环境变量 ``MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN``（指向 .so 文件）。
    2. ``extra_paths`` 中第一个存在的路径。
    3. 若干相对仓库根目录的默认构建产物路径。
    """

    env = os.environ.get("MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN")
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    candidates: list[Path] = []
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)

    here = Path(__file__).resolve()
    # model_optimizer/src/model_optimizer/ops -> repo root is parents[3]
    root = here.parents[3]
    candidates.extend(
        [
            root / "build" / "trt_plugins" / "siglip_ffn_fp8" / "libtrt_siglip_ffn_fp8_plugin.so",
            root / "csrc" / "build" / "trt_plugins" / "siglip_ffn_fp8" / "libtrt_siglip_ffn_fp8_plugin.so",
            root / "build_csrc" / "trt_plugins" / "siglip_ffn_fp8" / "libtrt_siglip_ffn_fp8_plugin.so",
        ]
    )
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def gpu_supports_fp8_trt() -> bool:
    """TensorRT FP8 插件路径在实践上需要 Ada（SM 8.9）及以上；Ampere 上建引擎会失败。"""

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)
