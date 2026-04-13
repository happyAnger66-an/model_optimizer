"""OrtEngine — ONNX Runtime Session 封装，接口对齐 TRT Engine。

用法与 ``infer/tensorrt/trt_torch.Engine`` 一致：

    engine = OrtEngine("vit.onnx", perf=True)
    out = engine(pixel_values=torch_tensor)   # → dict[str, torch.Tensor]
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import torch
from termcolor import colored


def _numpy_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """torch dtype → numpy dtype。"""
    _map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # ORT 不支持 bf16，转 fp32
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    if torch_dtype in _map:
        return _map[torch_dtype]
    raise TypeError(f"Unsupported torch dtype for ORT: {torch_dtype}")


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    """numpy dtype → torch dtype。"""
    _map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool,
    }
    kind = np.dtype(np_dtype)
    if kind.type in _map:
        return _map[kind.type]
    raise TypeError(f"Unsupported numpy dtype for torch: {np_dtype}")


def _select_providers(providers: list[str] | None = None) -> list[str]:
    """选择可用的 Execution Provider。

    默认顺序：TensorRT（若 ORT 构建带 TensorRTExecutionProvider）> CUDA > CPU。
    NVFP4 / FLOAT4E2M1FN（ONNX dtype 23）等通常需 TensorRT EP 或较新 ORT；纯 CUDA EP 常无法加载。
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if providers is not None:
        return [p for p in providers if p in available]
    preferred = [
        "TensorRTExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


class OrtEngine:
    """ONNX Runtime InferenceSession 封装，接口与 TRT ``Engine`` 对齐。

    - ``__call__(**kwargs) → dict[str, torch.Tensor]``
    - 输入：torch.Tensor（自动转 numpy）或 numpy.ndarray
    - 输出：dict，key 为 ONNX output name，value 为 torch.Tensor（GPU if available）
    - ``return_wrap``：可选的输出后处理函数（与 TRT Engine 一致）
    - ``perf=True``：记录推理耗时到 ``time_results``

    Args:
        onnx_path: ONNX 模型文件路径。
        return_wrap: 对输出 dict 的后处理函数。
        perf: 是否记录推理耗时。
        providers: ORT Execution Provider 列表。None 时使用默认顺序（含 TensorRT EP，若已安装）。
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        return_wrap: Callable[[dict[str, torch.Tensor]], Any] | None = None,
        perf: bool = False,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.file = onnx_path
        self.return_wrap = return_wrap
        self.perf = perf
        self.count = 0

        selected_providers = _select_providers(providers)
        self._use_cuda = "CUDAExecutionProvider" in selected_providers

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=selected_providers,
            )
        except Exception as exc:
            msg = str(exc)
            if "data type 23" in msg or "Invalid tensor data type" in msg:
                raise RuntimeError(
                    "ONNX Runtime 无法加载该 ONNX（常见原因：图内含 FLOAT4E2M1FN / NVFP4，即 dtype 23）。"
                    "请确认已安装带 TensorRTExecutionProvider 的 onnxruntime-gpu，且 TensorRT 版本支持该模型；"
                    "或升级 onnxruntime/onnx 至支持该类型的版本。"
                    "若仍失败，请改用 inference_mode=tensorrt 与预编译 .plan/.engine（Pi05TensorRTExecutor 路径）。"
                ) from exc
            raise

        # 缓存输入/输出元信息
        self.in_meta = [
            (inp.name, inp.shape, inp.type) for inp in self._session.get_inputs()
        ]
        self.out_meta = [
            (out.name, out.shape, out.type) for out in self._session.get_outputs()
        ]
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        if perf:
            self.time_results: dict[str, list[float]] = {"total": []}

        self.print()

    def print(self) -> None:
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= ORT Engine Detail =============")
        print(f"ONNX file: {self.file}")
        print(f"Providers: {self._session.get_providers()}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, (name, shape, dtype) in enumerate(self.in_meta):
            shape_str = "x".join(str(s) for s in shape) if shape else "scalar"
            print(f"   {ib}. {name}: {shape_str} [{dtype}]")
        print(f"Outputs: {len(self.out_meta)}")
        for ib, (name, shape, dtype) in enumerate(self.out_meta):
            shape_str = "x".join(str(s) for s in shape) if shape else "scalar"
            print(f"   {ib}. {name}: {shape_str} [{dtype}]")
        print("=============================================")

    def _to_numpy(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """将输入转为 numpy，bf16 先转 fp32。"""
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.bfloat16:
                x = x.float()
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        """将 numpy 输出转为 torch.Tensor，放到 GPU（如可用）。"""
        t = torch.from_numpy(x)
        if self._use_cuda and torch.cuda.is_available():
            t = t.cuda()
        return t

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        start_time = time.perf_counter()

        # 收集输入（支持位置参数和关键字参数，与 TRT Engine 一致）
        feeds: dict[str, np.ndarray] = {}
        for iarg, x in enumerate(args):
            if iarg >= len(self.in_meta):
                raise ValueError(
                    f"Too many positional inputs: got {len(args)}, expected <= {len(self.in_meta)}"
                )
            name = self._input_names[iarg]
            feeds[name] = self._to_numpy(x)

        for name, x in kwargs.items():
            if name == "return_list":
                continue
            feeds[name] = self._to_numpy(x)

        missing = [n for n in self._input_names if n not in feeds]
        if missing:
            raise ValueError(f"Missing required ORT inputs: {missing}")

        # 运行推理
        raw_outputs = self._session.run(self._output_names, feeds)

        # 构造输出 dict
        outputs: dict[str, torch.Tensor] = {}
        for name, arr in zip(self._output_names, raw_outputs):
            outputs[name] = self._to_torch(arr)

        end_time = time.perf_counter()
        if self.perf and self.count > 100:
            self.time_results["total"].append(end_time - start_time)

        if self.return_wrap:
            return self.return_wrap(outputs)
        return outputs
