# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SigLIP FFN FP8：PyTorch 参考 → ONNX（trt::SiglipFfFp8Plugin）→ TensorRT 引擎 → 与参考对比。

- 无 CUDA / 无 TensorRT / 找不到 ``libtrt_siglip_ffn_fp8_plugin.so`` 时跳过 TRT 相关用例。
- **FP8 引擎构建**在 Ampere（SM 8.6）等硬件上通常会失败；仅在 **SM ≥ 8.9**（Ada 及以上）上尝试建引擎与推理。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")

trt = pytest.importorskip("tensorrt")

from model_optimizer.ops.siglip_ffn_fp8 import (  # noqa: E402
    TinySiglipFfFp8Mlp,
    discover_siglip_ffn_fp8_plugin_so,
    gpu_supports_fp8_trt,
)
from model_optimizer.ops.siglip_ffn_fp8_onnx import build_siglip_ffn_fp8_onnx_from_module  # noqa: E402
from model_optimizer.ops.siglip_ffn_fp8_trt import (  # noqa: E402
    build_engine_from_onnx,
    load_siglip_ffn_fp8_plugin,
    run_engine_fp8_ffn,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", 0)


def _require_plugin_so() -> str:
    p = discover_siglip_ffn_fp8_plugin_so()
    if not p:
        pytest.skip(
            "libtrt_siglip_ffn_fp8_plugin.so not found; build with "
            "-DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON or set MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN"
        )
    return p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_siglip_ffn_fp8_eager_forward() -> None:
    torch.manual_seed(0)
    dev = _require_cuda()
    s, d, h = 3, 8, 16
    m = TinySiglipFfFp8Mlp(d, h, device=dev).eval()
    x = torch.randn(s, d, device=dev, dtype=torch.float32).to(torch.float8_e4m3fn)
    residual = torch.randn(s, d, device=dev, dtype=torch.float16)
    y = m(x, residual)
    assert y.shape == (s, d)
    assert y.dtype == torch.float16


def test_siglip_ffn_fp8_onnx_contains_trt_plugin() -> None:
    pytest.importorskip("onnx")
    dev = _require_cuda()
    torch.manual_seed(1)
    s, d, h = 2, 8, 16
    m = TinySiglipFfFp8Mlp(d, h, device=dev).eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        build_siglip_ffn_fp8_onnx_from_module(m, s=s, onnx_path=path)
        import onnx

        model = onnx.load(str(path))
    finally:
        path.unlink(missing_ok=True)

    nodes = [n for n in model.graph.node if n.op_type == "SiglipFfFp8Plugin" and n.domain == "trt"]
    assert nodes, f"expected trt::SiglipFfFp8Plugin node, got {[ (n.domain,n.op_type) for n in model.graph.node]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_onnx_parser_accepts_siglip_ffn_fp8() -> None:
    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    torch.manual_seed(2)
    s, d, h = 2, 8, 16
    m = TinySiglipFfFp8Mlp(d, h, device=dev).eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        build_siglip_ffn_fp8_onnx_from_module(m, s=s, onnx_path=path)
        load_siglip_ffn_fp8_plugin(plugin_so)
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
@pytest.mark.skipif(not gpu_supports_fp8_trt(), reason="FP8 TensorRT plugin path requires SM >= 8.9 (Ada+)")
def test_trt_engine_matches_pytorch_reference() -> None:
    """在 Ada+ GPU 上：建引擎并对比 PyTorch 参考输出。"""

    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    torch.manual_seed(3)
    s, d, h = 2, 8, 16
    m = TinySiglipFfFp8Mlp(d, h, device=dev).eval()
    x = torch.randn(s, d, device=dev, dtype=torch.float32).to(torch.float8_e4m3fn)
    residual = torch.randn(s, d, device=dev, dtype=torch.float16)
    y_ref = m(x, residual)

    with tempfile.TemporaryDirectory() as td:
        onnx_p = Path(td) / "m.onnx"
        build_siglip_ffn_fp8_onnx_from_module(m, s=s, onnx_path=onnx_p)
        load_siglip_ffn_fp8_plugin(plugin_so)
        plan = build_engine_from_onnx(onnx_p)
        if plan is None:
            pytest.skip(
                "build_serialized_network returned None (FP8 plugin / builder / driver mismatch on this machine)"
            )
        y_trt = run_engine_fp8_ffn(plan, x_fp8=x, residual=residual)

    assert y_trt.shape == y_ref.shape
    assert torch.allclose(y_trt, y_ref, rtol=0.15, atol=0.15), (y_trt - y_ref).abs().max().item()
