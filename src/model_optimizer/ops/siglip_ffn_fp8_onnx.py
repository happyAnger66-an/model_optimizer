# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""构建仅含 ``trt::SiglipMlpPlugin`` 的 ONNX 图，用于 TensorRT 插件与引擎的端到端测试。

与 :func:`model_optimizer.ops.siglip_mlp_plugin.siglip_mlp_plugin` / ONNX 导出节点一致；
权重以 initializer 固化，运行时输入仅为 ``x``（FP16）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

from model_optimizer.ops.siglip_mlp_plugin import ONNX_OPSET_VERSION, register_siglip_mlp_plugin_onnx_symbolic_functions

logger = logging.getLogger(__name__)

_SCHEMA_REGISTERED = False


class TinySiglipMlpTrt(nn.Module):
    """最小 SigLIP MLP（参数名与 :func:`build_siglip_mlp_trt_onnx_from_module` 约定一致）。"""

    def __init__(self, d: int, h: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        dev = device or torch.device("cpu")
        self.fc1_w = nn.Parameter(torch.randn(h, d, device=dev, dtype=dtype))
        self.fc1_b = nn.Parameter(torch.randn(h, device=dev, dtype=dtype))
        self.fc2_w = nn.Parameter(torch.randn(d, h, device=dev, dtype=dtype))
        self.fc2_b = nn.Parameter(torch.randn(d, device=dev, dtype=dtype))


def register_siglip_mlp_trt_onnx_schema() -> None:
    """注册 ``trt::SiglipMlpPlugin`` 的 ONNX schema（与 ``siglip_mlp_plugin`` 一致）。"""

    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    register_siglip_mlp_plugin_onnx_symbolic_functions()
    _SCHEMA_REGISTERED = True
    logger.debug("SiglipMlpPlugin ONNX schema registered via siglip_mlp_plugin.")


def build_siglip_mlp_trt_onnx_from_module(
    module: nn.Module,
    *,
    s: int,
    onnx_path: str | Path,
    act_id: int = 0,
) -> Path:
    """从带 ``fc1_w``/``fc1_b``/``fc2_w``/``fc2_b`` 的模块导出 ``SiglipMlpPlugin`` 单节点图（FP16）。"""

    register_siglip_mlp_trt_onnx_schema()
    m = module
    w1 = m.fc1_w.detach().float().cpu().numpy().astype(np.float16)  # type: ignore[attr-defined]
    b1 = m.fc1_b.detach().float().cpu().numpy().astype(np.float16)  # type: ignore[attr-defined]
    w2 = m.fc2_w.detach().float().cpu().numpy().astype(np.float16)  # type: ignore[attr-defined]
    b2 = m.fc2_b.detach().float().cpu().numpy().astype(np.float16)  # type: ignore[attr-defined]

    d = int(w1.shape[1])
    h = int(w1.shape[0])
    vi_x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [s, d])
    vo = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [s, d])

    node = helper.make_node(
        "SiglipMlpPlugin",
        inputs=["x", "fc1_w", "fc1_b", "fc2_w", "fc2_b"],
        outputs=["y"],
        domain="trt",
    )
    node.attribute.extend([helper.make_attribute("act_id", int(act_id))])

    graph = helper.make_graph(
        [node],
        "siglip_mlp_trt",
        [vi_x],
        [vo],
        initializer=[
            numpy_helper.from_array(w1, name="fc1_w"),
            numpy_helper.from_array(b1, name="fc1_b"),
            numpy_helper.from_array(w2, name="fc2_w"),
            numpy_helper.from_array(b2, name="fc2_b"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", int(ONNX_OPSET_VERSION)), helper.make_opsetid("trt", 1)],
    )
    onnx_path = Path(onnx_path)
    onnx.save(model, str(onnx_path))
    return onnx_path


def register_siglip_ffn_fp8_onnx_schema() -> None:
    """Deprecated: 请使用 :func:`register_siglip_mlp_trt_onnx_schema`。"""

    register_siglip_mlp_trt_onnx_schema()


def build_siglip_ffn_fp8_onnx_from_module(*_args: object, **_kwargs: object) -> Path:
    """Deprecated: 旧 FP8 单测入口已移除；请使用 :class:`TinySiglipMlpTrt` + :func:`build_siglip_mlp_trt_onnx_from_module`。"""

    raise RuntimeError(
        "build_siglip_ffn_fp8_onnx_from_module was removed. "
        "Use TinySiglipMlpTrt + build_siglip_mlp_trt_onnx_from_module (see tests/test_siglip_mlp_trt_e2e.py)."
    )
