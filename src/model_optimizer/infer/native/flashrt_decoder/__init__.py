"""Pi0.5 Thor decoder —— FlashRT 实现的仓内移植（不 import flash_rt 包）。

本包把 FlashRT 的 Pi0.5 Thor **decoder（AdaRMSNorm action expert，静态 FP8）** 实现
迁移进 ``model_optimizer`` 自有命名空间，作为 native denoise 后端的底层计算。

来源对应（FlashRT → 本包）：
- ``flash_rt/models/pi05/pipeline_thor.py`` → :mod:`.pipeline`
  （``decoder_forward`` / ``_decoder_forward_fp16`` / ``decoder_forward_calibrate``）
- ``flash_rt/hardware/thor/shared_primitives.py`` 中的 GPU 指针小工具 → :mod:`.cuda_helpers`
- ``flash_rt.flash_rt_kernels`` 编译扩展（``fvk``）→ 由 :mod:`.kernels` 按 ``.so`` 路径加载，
  **不 import ``flash_rt`` python 包**。

设计：计算编排（纯指针）属于本包；CUDA kernel（``fvk`` / FMHA ``.so``）由 FlashRT 的
``csrc`` 在 Thor(SM110) 上 cmake 构建后产出，本包按路径装载。详见
``docs/optimizer/ddup/native_decoder_implementation_todo.md`` §8。
"""

from __future__ import annotations

from .backend import FlashRtDecoderBackend
from .driver import (
    DecoderBuffers,
    DecoderWeights,
    Pi05ThorDecoderLoop,
    build_decoder_dims,
    fill_prefix_kv_from_trt,
)
from .kernels import load_kernels
from .pipeline import (
    decoder_forward,
    decoder_forward_calibrate,
)
from .precompute import (
    AdaRmsStyles,
    build_dec_rope,
    precompute_adarms_styles,
)
from .weights import (
    RepackedDecoderWeights,
    interleave_qk,
    quant_fp8,
    repack_decoder_weights,
    state_dict_getter,
)

__all__ = [
    "decoder_forward",
    "decoder_forward_calibrate",
    "load_kernels",
    "build_decoder_dims",
    "fill_prefix_kv_from_trt",
    "DecoderBuffers",
    "DecoderWeights",
    "Pi05ThorDecoderLoop",
    "FlashRtDecoderBackend",
    "AdaRmsStyles",
    "build_dec_rope",
    "precompute_adarms_styles",
    "RepackedDecoderWeights",
    "interleave_qk",
    "quant_fp8",
    "repack_decoder_weights",
    "state_dict_getter",
]
