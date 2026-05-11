# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP MLP：PyTorch eager → ONNX（``trt::SiglipMlpPlugin``）→ TensorRT 引擎 → 与参考对比。

- 无 CUDA / 无 TensorRT / 找不到 ``libtrt_siglip_mlp_plugin.so`` 时跳过 TRT 相关用例。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")

trt = pytest.importorskip("tensorrt")

from model_optimizer.ops.siglip_ffn_fp8 import discover_siglip_mlp_trt_plugin_so  # noqa: E402
from model_optimizer.ops.siglip_ffn_fp8_onnx import (  # noqa: E402
    TinySiglipMlpTrt,
    build_siglip_mlp_trt_onnx_from_module,
)
from model_optimizer.ops.siglip_ffn_fp8_trt import (  # noqa: E402
    build_engine_from_onnx,
    load_siglip_mlp_trt_plugin,
    run_engine_siglip_mlp,
)
from model_optimizer.ops.siglip_mlp import siglip_mlp_eager  # noqa: E402


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", 0)


def _require_plugin_so() -> str:
    p = discover_siglip_mlp_trt_plugin_so()
    if not p:
        pytest.skip(
            "libtrt_siglip_mlp_plugin.so not found; build with "
            "-DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON or set MODEL_OPTIMIZER_TRT_SIGLIP_MLP_PLUGIN"
        )
    return p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_siglip_mlp_trt_onnx_contains_plugin() -> None:
    pytest.importorskip("onnx")
    dev = _require_cuda()
    torch.manual_seed(1)
    s, d, h = 2, 8, 16
    m = TinySiglipMlpTrt(d, h, device=dev).eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        build_siglip_mlp_trt_onnx_from_module(m, s=s, onnx_path=path, act_id=0)
        import onnx

        model = onnx.load(str(path))
    finally:
        path.unlink(missing_ok=True)

    nodes = [n for n in model.graph.node if n.op_type == "SiglipMlpPlugin" and n.domain == "trt"]
    assert nodes, f"expected trt::SiglipMlpPlugin node, got {[(n.domain, n.op_type) for n in model.graph.node]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_onnx_parser_accepts_siglip_mlp_plugin() -> None:
    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    torch.manual_seed(2)
    s, d, h = 2, 8, 16
    m = TinySiglipMlpTrt(d, h, device=dev).eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        build_siglip_mlp_trt_onnx_from_module(m, s=s, onnx_path=path, act_id=0)
        load_siglip_mlp_trt_plugin(plugin_so)
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        ok = parser.parse_from_file(str(path))
        if not ok:
            for i in range(parser.num_errors):
                print(parser.get_error(i))
        assert ok, "OnnxParser failed"
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_engine_matches_pytorch_eager() -> None:
    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    torch.manual_seed(3)
    s, d, h = 2, 8, 16
    m = TinySiglipMlpTrt(d, h, device=dev).eval()
    x = torch.randn(s, d, device=dev, dtype=torch.float16)
    y_ref = siglip_mlp_eager(x, m.fc1_w, m.fc1_b, m.fc2_w, m.fc2_b, 0)

    with tempfile.TemporaryDirectory() as td:
        onnx_p = Path(td) / "m.onnx"
        build_siglip_mlp_trt_onnx_from_module(m, s=s, onnx_path=onnx_p, act_id=0)
        load_siglip_mlp_trt_plugin(plugin_so)
        plan = build_engine_from_onnx(onnx_p)
        if plan is None:
            pytest.skip("build_serialized_network returned None (TensorRT / plugin mismatch on this machine)")
        y_trt = run_engine_siglip_mlp(plan, x=x)

    assert y_trt.shape == y_ref.shape
    assert torch.allclose(y_trt, y_ref, rtol=2e-2, atol=2e-2), (y_trt - y_ref).abs().max().item()
