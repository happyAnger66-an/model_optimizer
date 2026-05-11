# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP FFN FP8 融合路径的 PyTorch 参考实现（与 ``SiglipFfFp8Plugin`` / FlashRT 数学顺序对齐）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SIGLIP_FF8_FLASHRT_CAST_WARNED = False


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


class SiglipFfFp8FlashrtMlpWrapper(nn.Module):
    """包装 HF ``SiglipMLP``：默认走 ``inner``；导出 ONNX 为 ``trt::SiglipFfFp8FlashrtPlugin`` 时走 :func:`siglip_ffn_fp8_flashrt_plugin`。

    权重需已为 ``torch.float8_e4m3fn``（通常来自 ``quantize_model`` + FP8 配置）；``unit_scale`` 优先读取 ``inner.unit_scale``，
    否则使用可学习/默认标量 ``[1.0]``（生产环境请与 FlashRT 标定一致）。
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        us = getattr(inner, "unit_scale", None)
        if isinstance(us, torch.Tensor):
            self.register_buffer("_unit_scale_buf", us.detach().float().view(-1).clone())
        else:
            self.register_buffer("_unit_scale_buf", torch.tensor([1.0], dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_trt = os.environ.get("MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not use_trt:
            return self.inner(hidden_states)

        m = self.inner
        w1, b1, w2, b2 = m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            logger.warning(
                "MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT is set but MLP weights are not float8_e4m3fn; "
                "falling back to inner MLP forward. Run quantize_model with FP8 cfg on fc1/fc2 first."
            )
            return m(hidden_states)

        up_w_fp8 = w1.t().contiguous()
        down_w_fp8 = w2.t().contiguous()
        res = hidden_states.to(torch.float16)
        global _SIGLIP_FF8_FLASHRT_CAST_WARNED
        if hidden_states.dtype != torch.float8_e4m3fn:
            if not _SIGLIP_FF8_FLASHRT_CAST_WARNED:
                logger.warning(
                    "FFN FP8 TRT export: hidden_states are not float8_e4m3fn; using cast-to-fp8 (dev fallback). "
                    "For production, feed true quantized activations matching FlashRT."
                )
                _SIGLIP_FF8_FLASHRT_CAST_WARNED = True
            x_fp8 = hidden_states.to(torch.float16).to(torch.float8_e4m3fn)
        else:
            x_fp8 = hidden_states

        us = getattr(m, "unit_scale", None)
        unit = us if isinstance(us, torch.Tensor) else self._unit_scale_buf
        if unit.ndim == 0:
            unit = unit.view(1)
        unit = unit.to(device=hidden_states.device, dtype=torch.float32).reshape(-1)

        from .siglip_ffn_fp8_flashrt_export import siglip_ffn_fp8_flashrt_plugin

        return siglip_ffn_fp8_flashrt_plugin(
            x_fp8,
            res,
            up_w_fp8,
            down_w_fp8,
            b1.to(torch.float16),
            b2.to(torch.float16),
            unit,
            1.0,
            1.0,
        )


def patch_vision_siglip_ffn_fp8_flashrt_custom_op(vision_tower: nn.Module, *, enabled: bool) -> int:
    """将各 encoder layer 的 ``mlp`` 替换为 :class:`SiglipFfFp8FlashrtMlpWrapper`（``enabled`` 时）。"""

    if not enabled:
        return 0

    from .siglip_mlp import _iter_siglip_encoder_layers

    n = 0
    for layer in _iter_siglip_encoder_layers(vision_tower):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if isinstance(mlp, SiglipFfFp8FlashrtMlpWrapper):
            continue
        layer.mlp = SiglipFfFp8FlashrtMlpWrapper(mlp)
        n += 1
    if n:
        logger.info("SigLIP FFN FP8 (FlashRT TRT) wrapper: patched %d encoder layer(s).", n)
    else:
        logger.warning("SigLIP FFN FP8 FlashRT enabled but no encoder layers found under vision_tower.")
    return n


def discover_siglip_mlp_trt_plugin_so(extra_paths: Iterable[str | Path] | None = None) -> str | None:
    """解析 ``libtrt_siglip_mlp_plugin.so``（SigLIP MLP TensorRT 插件）路径。

    优先级：

    1. ``MODEL_OPTIMIZER_TRT_SIGLIP_MLP_PLUGIN``（指向 .so）。
    2. ``MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN``（历史环境变量，兼容旧文档）。
    3. ``extra_paths`` 中第一个存在的路径。
    4. 相对仓库根目录的默认构建产物路径（同时尝试新、旧 .so 文件名）。
    """

    for key in ("MODEL_OPTIMIZER_TRT_SIGLIP_MLP_PLUGIN", "MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN"):
        env = os.environ.get(key)
        if env and Path(env).is_file():
            return str(Path(env).resolve())

    candidates: list[Path] = []
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)

    here = Path(__file__).resolve()
    root = here.parents[3]
    sub = Path("trt_plugins") / "siglip_ffn_fp8"
    candidates.extend(
        [
            root / "build" / sub / "libtrt_siglip_mlp_plugin.so",
            root / "csrc" / "build" / sub / "libtrt_siglip_mlp_plugin.so",
            root / "build_csrc" / sub / "libtrt_siglip_mlp_plugin.so",
            root / "build" / sub / "libtrt_siglip_ffn_fp8_plugin.so",
            root / "csrc" / "build" / sub / "libtrt_siglip_ffn_fp8_plugin.so",
            root / "build_csrc" / sub / "libtrt_siglip_ffn_fp8_plugin.so",
        ]
    )
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def discover_siglip_ffn_fp8_plugin_so(extra_paths: Iterable[str | Path] | None = None) -> str | None:
    """Deprecated: 请使用 :func:`discover_siglip_mlp_trt_plugin_so`。"""

    return discover_siglip_mlp_trt_plugin_so(extra_paths)


def discover_siglip_ffn_fp8_flashrt_plugin_so(extra_paths: Iterable[str | Path] | None = None) -> str | None:
    """解析 ``libtrt_siglip_ffn_fp8_flashrt_plugin.so``（FlashRT 桥接的 FP8 FFN TensorRT 插件）。

    需先在 ``third_party/FlashRT`` 构建出 ``flash_rt/libflash_rt_trt_bridge.a``，再编译
    ``csrc`` 的 ``trt_siglip_ffn_fp8_flashrt_plugin`` 目标。

    环境变量 ``MODEL_OPTIMIZER_TRT_SIGLIP_FF8_FLASHRT_PLUGIN`` 可显式指向 .so。
    """

    env = os.environ.get("MODEL_OPTIMIZER_TRT_SIGLIP_FF8_FLASHRT_PLUGIN")
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    candidates: list[Path] = []
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)

    here = Path(__file__).resolve()
    root = here.parents[3]
    sub = Path("trt_plugins") / "siglip_ffn_fp8"
    candidates.extend(
        [
            root / "build" / sub / "libtrt_siglip_ffn_fp8_flashrt_plugin.so",
            root / "csrc" / "build" / sub / "libtrt_siglip_ffn_fp8_flashrt_plugin.so",
            root / "build_csrc" / sub / "libtrt_siglip_ffn_fp8_flashrt_plugin.so",
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
