"""load_infer_bundle 参数校验。"""

from __future__ import annotations

from .config import Args


def validate_infer_bundle_args(args: Args) -> None:
    modes = [
        bool(args.compare_mode),
        bool(args.ptq_compare),
        bool(getattr(args, "ptq_trt_compare", False)),
        bool(getattr(args, "ort_compare", False)),
        bool(getattr(args, "trt_ort_compare", False)),
        bool(getattr(args, "trt_trt_compare", False)),
    ]
    if sum(1 for x in modes if x) > 1:
        raise ValueError(
            "compare_mode / ptq_compare / ptq_trt_compare / ort_compare / "
            "trt_ort_compare / trt_trt_compare 互斥，请勿同时开启。"
        )
    if (args.ptq_compare or getattr(args, "ptq_trt_compare", False)) and args.inference_mode != "pytorch":
        raise ValueError("ptq_compare / ptq_trt_compare 仅支持 inference_mode=pytorch。")
    if getattr(args, "trt_trt_compare", False) and args.inference_mode != "tensorrt":
        raise ValueError(
            "trt_trt_compare=True 时必须设置 --inference-mode tensorrt（双路均为 TensorRT 引擎）。"
        )
