# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TensorRT：加载 ``libtrt_siglip_mlp_plugin.so``，自 ONNX（``trt::SiglipMlpPlugin``）构建引擎并在 GPU 上推理。"""

from __future__ import annotations

import ctypes
from pathlib import Path

import tensorrt as trt
import torch


def load_siglip_mlp_trt_plugin(plugin_so: str) -> None:
    ctypes.CDLL(str(plugin_so), ctypes.RTLD_GLOBAL)


def build_engine_from_onnx(
    onnx_path: str | Path,
    *,
    workspace_bytes: int = 1 << 30,
    logger: trt.ILogger | None = None,
) -> bytes | None:
    """解析 ONNX 并序列化引擎；失败时返回 ``None``。"""

    logger = logger or trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    path = str(onnx_path)
    if not parser.parse_from_file(path):
        for i in range(parser.num_errors):
            logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
        return None

    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        return None
    return bytes(plan)


def run_engine_siglip_mlp(
    engine_bytes: bytes,
    *,
    x: torch.Tensor,
    logger: trt.ILogger | None = None,
) -> torch.Tensor:
    """对仅含 ``x`` 输入与 ``y`` 输出的 ``SiglipMlpPlugin`` 引擎执行一次推理。"""

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
    out_tensor: torch.Tensor | None = None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dt = engine.get_tensor_dtype(name)
        shape = tuple(ctx.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            if name not in bindings:
                raise KeyError(f"Missing input binding {name}")
            ctx.set_tensor_address(name, bindings[name].data_ptr())
        elif mode == trt.TensorIOMode.OUTPUT:
            if name != "y":
                raise NotImplementedError(f"Unexpected output {name!r}; this helper only supports y")
            if dt == trt.DataType.HALF:
                tdt = torch.float16
            elif dt == trt.DataType.FLOAT:
                tdt = torch.float32
            elif dt == trt.DataType.BF16:
                tdt = torch.bfloat16
            else:
                raise TypeError(f"Unsupported output dtype {dt}")
            out_tensor = torch.empty(shape, dtype=tdt, device=x.device)
            ctx.set_tensor_address(name, out_tensor.data_ptr())
        else:
            raise RuntimeError(f"Unexpected tensor mode for {name}")

    if out_tensor is None:
        raise RuntimeError("No output tensor y allocated")
    ok = ctx.execute_async_v3(stream.cuda_stream)
    if not ok:
        raise RuntimeError("execute_async_v3 returned False")
    stream.synchronize()
    return out_tensor


# Backward-compatible aliases (historical names).
def load_siglip_ffn_fp8_plugin(plugin_so: str) -> None:
    load_siglip_mlp_trt_plugin(plugin_so)


def run_engine_fp8_ffn(
    engine_bytes: bytes,
    *,
    x_fp8: torch.Tensor,
    residual: torch.Tensor,
    logger: trt.ILogger | None = None,
) -> torch.Tensor:
    _ = (engine_bytes, x_fp8, residual, logger)
    raise RuntimeError(
        "run_engine_fp8_ffn was removed. Use run_engine_siglip_mlp(engine_bytes, x=...) with FP16 x."
    )
