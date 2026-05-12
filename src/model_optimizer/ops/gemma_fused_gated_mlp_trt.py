# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""TensorRT：加载 ``libtrt_gemma_fused_gated_mlp_plugin.so``，从 ONNX 构建引擎并执行 GemmaFusedGatedMlp 插件。"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable

import tensorrt as trt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PLUGIN_REL_PATHS = (
    "build/trt_plugins/gemma_fused_gated_mlp/libtrt_gemma_fused_gated_mlp_plugin.so",
    "csrc/build/trt_plugins/gemma_fused_gated_mlp/libtrt_gemma_fused_gated_mlp_plugin.so",
    "build_csrc/trt_plugins/gemma_fused_gated_mlp/libtrt_gemma_fused_gated_mlp_plugin.so",
)


def _plugin_candidate_paths(extra_paths: Iterable[str | Path] | None = None) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN")
    if env:
        out.append(Path(env).expanduser())
    if extra_paths:
        out.extend(Path(p) for p in extra_paths)
    for rel in _DEFAULT_PLUGIN_REL_PATHS:
        out.append(_REPO_ROOT / rel)
    return out


def explain_gemma_fused_gated_mlp_plugin_discovery(extra_paths: Iterable[str | Path] | None = None) -> str:
    """未找到插件时用于日志：列出环境变量与已检查路径是否存在。"""

    lines: list[str] = []
    env = os.environ.get("MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN")
    lines.append(f"MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN={env!r}")
    lines.append(f"repo_root={_REPO_ROOT}")
    for p in _plugin_candidate_paths(extra_paths):
        exists = p.is_file()
        lines.append(f"  {'OK ' if exists else 'MISS'} {p.resolve() if p.exists() else p}")
    lines.append(
        "Fix: build csrc with -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON and export "
        "MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN=/abs/path/to/libtrt_gemma_fused_gated_mlp_plugin.so"
    )
    return "\n".join(lines)


def discover_gemma_fused_gated_mlp_plugin_so(extra_paths: Iterable[str | Path] | None = None) -> str | None:
    """解析 ``libtrt_gemma_fused_gated_mlp_plugin.so`` 路径。

    优先级：

    1. 环境变量 ``MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN``（指向 .so）。
    2. ``extra_paths`` 中第一个存在的路径。
    3. 相对仓库根目录的常见 CMake 构建产物路径。
    """

    for p in _plugin_candidate_paths(extra_paths):
        if p.is_file():
            return str(p.resolve())
    return None


def load_gemma_fused_gated_mlp_plugin(plugin_so: str) -> None:
    """加载插件 ``.so``（``RTLD_GLOBAL``）。

    TensorRT 10+ 在 ``IPluginRegistry::loadLibrary`` / trtexec ``--dynamicPlugins`` 等路径下会
    ``dlsym(getCreators)``；插件库须导出 ``getCreators`` / ``setLoggerFinder``（见
    ``csrc/trt_plugins/gemma_fused_gated_mlp/gemma_fused_gated_mlp_plugin.cpp``）。
    """

    ctypes.CDLL(str(plugin_so), ctypes.RTLD_GLOBAL)


def init_trt_plugins_after_load(logger: trt.ILogger | None = None) -> None:
    """在 ``CDLL`` 加载插件 .so 之后调用，注册 TensorRT 插件入口。"""

    logger = logger or trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")


def build_gemma_fused_gated_mlp_engine_from_onnx(
    onnx_path: str | Path,
    *,
    workspace_bytes: int = 1 << 28,
    logger: trt.ILogger | None = None,
    use_bf16: bool = False,
    diagnostics_out: list[str] | None = None,
) -> bytes | None:
    """解析 ONNX（含 ``trt::GemmaFusedGatedMlp``）并序列化引擎；失败时返回 ``None``。

    若传入 ``diagnostics_out``，会把 ONNX 解析错误、建引擎失败等说明追加到该列表（便于打印到 stderr）。
    """

    def diag(msg: str) -> None:
        if diagnostics_out is not None:
            diagnostics_out.append(msg)

    logger = logger or trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    path = str(onnx_path)
    if not parser.parse_from_file(path):
        diag("OnnxParser failed:")
        for i in range(parser.num_errors):
            err = str(parser.get_error(i))
            diag(f"  [{i}] {err}")
            logger.log(trt.Logger.ERROR, err)
        return None

    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    bf16_flag = getattr(trt.BuilderFlag, "BF16", None)
    if use_bf16 and bf16_flag is not None:
        cfg.set_flag(bf16_flag)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        diag(
            "IBuilder.build_serialized_network returned None (GPU/driver/plugin incompatibility, "
            "insufficient workspace, or unsupported layer for this TensorRT version)."
        )
        return None
    return bytes(plan)


def _trt_dtype_to_torch(dt: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
    }
    if dt not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype for Gemma MLP output: {dt}")
    return mapping[dt]


def run_gemma_fused_gated_mlp_engine(
    engine_bytes: bytes,
    *,
    x: torch.Tensor,
    gate_up_weight: torch.Tensor | None = None,
    down_weight: torch.Tensor | None = None,
    logger: trt.ILogger | None = None,
) -> torch.Tensor:
    """执行一次推理。

    - **插件 v1**（三输入 ONNX）：需提供 ``x``、``gate_up_weight``、``down_weight``（CUDA）。
    - **插件 v2**（权重烘焙进引擎）：仅需 ``x``；权重参数必须为 ``None``。
    """

    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")

    logger = logger or trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine failed")
    ctx = engine.create_execution_context()
    stream = torch.cuda.current_stream()

    bindings: dict[str, torch.Tensor] = {"x": x.contiguous()}
    if gate_up_weight is not None and down_weight is not None:
        if not gate_up_weight.is_cuda or not down_weight.is_cuda:
            raise ValueError("gate_up_weight and down_weight must be CUDA tensors when provided")
        bindings["gate_up_weight"] = gate_up_weight.contiguous()
        bindings["down_weight"] = down_weight.contiguous()
    elif gate_up_weight is not None or down_weight is not None:
        raise ValueError("Provide both gate_up_weight and down_weight, or neither (baked-weights engine).")
    out_tensor: torch.Tensor | None = None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            if name not in bindings:
                all_names = [engine.get_tensor_name(j) for j in range(engine.num_io_tensors)]
                raise KeyError(
                    f"TensorRT input tensor {name!r} not in bindings {sorted(bindings)}; "
                    f"all_io_tensors={all_names}"
                )
            t_in = bindings[name]
            shape = tuple(int(d) for d in t_in.shape)
            if hasattr(ctx, "set_input_shape"):
                if not ctx.set_input_shape(name, shape):
                    raise RuntimeError(f"set_input_shape failed for {name} {shape}")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dt = engine.get_tensor_dtype(name)
        if mode == trt.TensorIOMode.OUTPUT:
            if name != "y":
                raise NotImplementedError(f"Unexpected output {name!r}; expected y")
            shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            tdt = _trt_dtype_to_torch(dt)
            out_tensor = torch.empty(shape, dtype=tdt, device=x.device)
        elif mode != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Unexpected tensor mode for {name}")

    if out_tensor is None:
        raise RuntimeError("No output tensor y allocated")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            ctx.set_tensor_address(name, bindings[name].data_ptr())
        elif mode == trt.TensorIOMode.OUTPUT and name == "y":
            ctx.set_tensor_address(name, out_tensor.data_ptr())

    ok = ctx.execute_async_v3(stream.cuda_stream)
    if not ok:
        raise RuntimeError("execute_async_v3 returned False")
    stream.synchronize()
    return out_tensor
