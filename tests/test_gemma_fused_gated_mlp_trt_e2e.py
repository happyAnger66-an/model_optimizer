# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""GemmaFusedGatedMlp：PyTorch 参考 → ONNX（trt::GemmaFusedGatedMlp）→ TensorRT 引擎 → 与参考对比。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnx")

trt = pytest.importorskip("tensorrt")

from model_optimizer.ops.gemma_fused_gated_mlp import gemma_fused_gated_mlp_eager  # noqa: E402
from model_optimizer.ops.gemma_fused_gated_mlp_plugin import (  # noqa: E402
    gemma_fused_gated_mlp_plugin,
    register_gemma_fused_gated_mlp_onnx_symbolic_functions,
)
from model_optimizer.ops.gemma_fused_gated_mlp_trt import (  # noqa: E402
    build_gemma_fused_gated_mlp_engine_from_onnx,
    discover_gemma_fused_gated_mlp_plugin_so,
    init_trt_plugins_after_load,
    load_gemma_fused_gated_mlp_plugin,
    run_gemma_fused_gated_mlp_engine,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", 0)


def _require_plugin_so() -> str:
    p = discover_gemma_fused_gated_mlp_plugin_so()
    if not p:
        pytest.skip(
            "libtrt_gemma_fused_gated_mlp_plugin.so not found; build with "
            "-DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON or set MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN"
        )
    return p


class _ExportMlp(nn.Module):
    """三输入 ONNX：与 TRT 插件绑定名 ``x`` / ``gate_up_weight`` / ``down_weight`` 一致。"""

    def forward(self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
        return gemma_fused_gated_mlp_plugin(x, gate_up_weight, down_weight, 1)


def _export_onnx_fp16(path: Path, dev: torch.device, b: int, t: int, h: int, inter: int) -> None:
    register_gemma_fused_gated_mlp_onnx_symbolic_functions()
    torch.manual_seed(42)
    m = _ExportMlp().to(device=dev, dtype=torch.float16).eval()
    x = torch.randn(b, t, h, device=dev, dtype=torch.float16)
    w_gu = torch.randn(2 * inter, h, device=dev, dtype=torch.float16)
    w_d = torch.randn(h, inter, device=dev, dtype=torch.float16)
    torch.onnx.export(
        m,
        (x, w_gu, w_d),
        str(path),
        input_names=["x", "gate_up_weight", "down_weight"],
        output_names=["y"],
        opset_version=19,
        dynamo=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma_fused_gated_mlp_onnx_contains_trt_plugin() -> None:
    dev = _require_cuda()
    b, t, h, inter = 1, 3, 16, 24
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        _export_onnx_fp16(path, dev, b, t, h, inter)
        import onnx

        model = onnx.load(str(path))
    finally:
        path.unlink(missing_ok=True)

    nodes = [n for n in model.graph.node if n.op_type == "GemmaFusedGatedMlp" and n.domain == "trt"]
    assert nodes, f"expected trt::GemmaFusedGatedMlp node, got {[(n.domain, n.op_type) for n in model.graph.node]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_onnx_parser_accepts_gemma_fused_gated_mlp() -> None:
    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    b, t, h, inter = 1, 2, 16, 24
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.onnx"
        _export_onnx_fp16(path, dev, b, t, h, inter)
        load_gemma_fused_gated_mlp_plugin(plugin_so)
        init_trt_plugins_after_load()
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        ok = parser.parse_from_file(str(path))
        if not ok:
            for i in range(parser.num_errors):
                print(parser.get_error(i))
        assert ok, "OnnxParser failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_engine_matches_pytorch_gemma_fused_gated_mlp_fp16() -> None:
    """加载插件、建 FP16 引擎，对比 ``gemma_fused_gated_mlp_eager`` 参考输出。"""

    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    b, t, h, inter = 2, 5, 32, 48
    torch.manual_seed(7)
    x = torch.randn(b, t, h, device=dev, dtype=torch.float16)
    w_gu = torch.randn(2 * inter, h, device=dev, dtype=torch.float16)
    w_d = torch.randn(h, inter, device=dev, dtype=torch.float16)
    y_ref = gemma_fused_gated_mlp_eager(x, w_gu, w_d, 1)

    with tempfile.TemporaryDirectory() as td:
        onnx_p = Path(td) / "m.onnx"
        _export_onnx_fp16(onnx_p, dev, b, t, h, inter)
        load_gemma_fused_gated_mlp_plugin(plugin_so)
        init_trt_plugins_after_load()
        plan = build_gemma_fused_gated_mlp_engine_from_onnx(onnx_p, use_bf16=False)
        if plan is None:
            pytest.skip("build_serialized_network returned None (TensorRT / ONNX / plugin mismatch on this machine)")
        y_trt = run_gemma_fused_gated_mlp_engine(plan, x=x, gate_up_weight=w_gu, down_weight=w_d)

    assert y_trt.shape == y_ref.shape
    assert y_trt.dtype == y_ref.dtype
    assert torch.allclose(y_trt, y_ref, rtol=2e-2, atol=2e-2), (y_trt - y_ref).abs().max().item()
