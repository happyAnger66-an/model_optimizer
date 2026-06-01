"""推理后端策略集合。"""

from __future__ import annotations

from .base import InferBackend, PredictionPack
from .flashrt import SingleFlashRtBackend
from .native import SingleNativeBackend
from .onnxrt import PtOrtCompareBackend, SingleOnnxRTBackend
from .pt_flashrt_compare import PtFlashRtCompareBackend
from .pt_ptq_compare import PtPtqCompareBackend
from .pt_trt_compare import PtTrtCompareBackend
from .ptq_trt_compare import PtqTrtCompareBackend
from .pytorch import SinglePyTorchBackend
from .tensorrt import SingleTensorRTBackend

__all__ = [
    "InferBackend",
    "PredictionPack",
    "SinglePyTorchBackend",
    "SingleFlashRtBackend",
    "SingleNativeBackend",
    "SingleTensorRTBackend",
    "SingleOnnxRTBackend",
    "PtTrtCompareBackend",
    "PtFlashRtCompareBackend",
    "PtPtqCompareBackend",
    "PtqTrtCompareBackend",
    "PtOrtCompareBackend",
]
