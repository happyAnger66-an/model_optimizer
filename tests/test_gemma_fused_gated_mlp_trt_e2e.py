# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""GemmaFusedGatedMlp：PyTorch 参考 → ONNX（trt::GemmaFusedGatedMlp）→ TensorRT 引擎 → 与参考对比。

未找到 ``libtrt_gemma_fused_gated_mlp_plugin.so`` 或建引擎失败时，会向 **stderr** 打印 ``[GEMMA_TRT_E2E]`` 失败说明（默认仍 ``pytest.skip`` 以免无插件环境整页红）。

若希望在 CI 中 **强制失败**（无插件即红），设置环境变量::

    MODEL_OPTIMIZER_REQUIRE_GEMMA_FUSED_GATED_MLP_TRT=1

查看失败/跳过原因::

    pytest tests/test_gemma_fused_gated_mlp_trt_e2e.py -rs          # skip 摘要
    pytest tests/test_gemma_fused_gated_mlp_trt_e2e.py -s           # 实时 stderr（含路径列表）
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
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
    explain_gemma_fused_gated_mlp_plugin_discovery,
    init_trt_plugins_after_load,
    load_gemma_fused_gated_mlp_plugin,
    run_gemma_fused_gated_mlp_engine,
)

_LOG_PREFIX = "[GEMMA_TRT_E2E]"


def _require_trt_plugin_strict() -> bool:
    v = os.environ.get("MODEL_OPTIMIZER_REQUIRE_GEMMA_FUSED_GATED_MLP_TRT", "")
    return v.strip().lower() in ("1", "true", "yes")


def _emit_failure(what: str, detail: str = "", *, max_warn_len: int = 1200) -> None:
    """打到 stderr，并 ``warnings.warn``，便于无 ``-s`` 时仍出现在 pytest 汇总里。"""
    line = f"{_LOG_PREFIX} FAILURE: {what}"
    if detail:
        line += f"\n{detail}"
    print(line, file=sys.stderr, flush=True)
    wmsg = f"{_LOG_PREFIX} {what}"
    if detail:
        wmsg += " — " + (detail if len(detail) <= max_warn_len else detail[:max_warn_len] + "...")
    warnings.warn(wmsg, UserWarning, stacklevel=2)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", 0)


def _require_plugin_so() -> str:
    p = discover_gemma_fused_gated_mlp_plugin_so()
    if p:
        return p
    explain = explain_gemma_fused_gated_mlp_plugin_discovery()
    _emit_failure("libtrt_gemma_fused_gated_mlp_plugin.so not found", explain)
    msg = (
        "libtrt_gemma_fused_gated_mlp_plugin.so not found; build with "
        "-DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON or set MODEL_OPTIMIZER_TRT_GEMMA_FUSED_GATED_MLP_PLUGIN. "
        "See stderr for path list. Set MODEL_OPTIMIZER_REQUIRE_GEMMA_FUSED_GATED_MLP_TRT=1 to fail instead of skip."
    )
    if _require_trt_plugin_strict():
        pytest.fail(msg)
    pytest.skip(msg)


class _ExportMlp(nn.Module):
    """三输入 ONNX：与 TRT 插件绑定名 ``x`` / ``gate_up_weight`` / ``down_weight`` 一致。"""

    def forward(self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
        return gemma_fused_gated_mlp_plugin(x, gate_up_weight, down_weight, 1)


def _stable_fp16_weights(
    dev: torch.device, b: int, t: int, h: int, inter: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """有界 fp16 随机权重，避免全零/极端溢出；与导出 ONNX、TRT 推理必须使用同一组张量。"""

    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    x = (torch.randn(b, t, h, device=dev, dtype=torch.float16, generator=g) * 0.12).clamp(-1.5, 1.5)
    w_gu = (torch.randn(2 * inter, h, device=dev, dtype=torch.float16, generator=g) * 0.08).clamp(-0.6, 0.6)
    w_d = (torch.randn(h, inter, device=dev, dtype=torch.float16, generator=g) * 0.08).clamp(-0.6, 0.6)
    return x, w_gu, w_d


def _export_onnx_fp16(
    path: Path,
    dev: torch.device,
    x: torch.Tensor,
    w_gu: torch.Tensor,
    w_d: torch.Tensor,
) -> None:
    """必须用与 ``y_ref`` / TRT 推理**完全相同**的 ``(x, w_gu, w_d)`` 导出，否则 ONNX 与运行时权重不一致。"""
    register_gemma_fused_gated_mlp_onnx_symbolic_functions()
    m = _ExportMlp().to(device=dev, dtype=torch.float16).eval()
    torch.onnx.export(
        m,
        (x, w_gu, w_d),
        str(path),
        input_names=["x", "gate_up_weight", "down_weight"],
        output_names=["y"],
        opset_version=19,
        dynamo=False,
    )


def _reference_fp32(x: torch.Tensor, w_gu: torch.Tensor, w_d: torch.Tensor) -> torch.Tensor:
    """fp32 参考，避免 fp16 下中间量退化为全零却仍通过 atol 的退化情况。"""
    return gemma_fused_gated_mlp_eager(
        x.float(),
        w_gu.float(),
        w_d.float(),
        1,
    )


def _assert_trt_matches_reference(y_trt: torch.Tensor, y_ref_fp32: torch.Tensor) -> None:
    assert torch.isfinite(y_ref_fp32).all(), "fp32 reference has non-finite values"
    lim = y_ref_fp32.abs().max().item()
    assert lim > 1e-4, f"degenerate fp32 reference (max|y|={lim}); use non-trivial inputs"
    assert torch.isfinite(y_trt).all(), (
        f"TRT output non-finite (nan/inf count={(~torch.isfinite(y_trt)).sum().item()}); check plugin / bindings"
    )
    y_trt_f = y_trt.float()
    assert torch.allclose(y_trt_f, y_ref_fp32, rtol=0.08, atol=0.08), (
        f"max abs err={(y_trt_f - y_ref_fp32).abs().max().item():.6g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma_fused_gated_mlp_onnx_contains_trt_plugin() -> None:
    dev = _require_cuda()
    b, t, h, inter = 1, 3, 16, 24
    x, w_gu, w_d = _stable_fp16_weights(dev, b, t, h, inter, seed=10001)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)
    try:
        _export_onnx_fp16(path, dev, x, w_gu, w_d)
        import onnx

        model = onnx.load(str(path))
    finally:
        path.unlink(missing_ok=True)

    nodes = [n for n in model.graph.node if n.op_type == "GemmaFusedGatedMlp" and n.domain == "trt"]
    assert nodes, f"expected trt::GemmaFusedGatedMlp node, got {[(n.domain, n.op_type) for n in model.graph.node]}"

    if not discover_gemma_fused_gated_mlp_plugin_so():
        _emit_failure(
            "ONNX graph OK but TRT plugin .so is missing — parser/engine e2e tests will be skipped",
            explain_gemma_fused_gated_mlp_plugin_discovery(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_onnx_parser_accepts_gemma_fused_gated_mlp() -> None:
    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    b, t, h, inter = 1, 2, 16, 24
    x, w_gu, w_d = _stable_fp16_weights(dev, b, t, h, inter, seed=10002)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.onnx"
        _export_onnx_fp16(path, dev, x, w_gu, w_d)
        load_gemma_fused_gated_mlp_plugin(plugin_so)
        init_trt_plugins_after_load()
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        ok = parser.parse_from_file(str(path))
        if not ok:
            lines = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            _emit_failure("TensorRT OnnxParser rejected the ONNX", "\n".join(lines))
            msg = "OnnxParser failed (details on stderr)"
            if _require_trt_plugin_strict():
                pytest.fail(msg)
            pytest.skip(msg)
        assert ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_engine_matches_pytorch_gemma_fused_gated_mlp_fp16() -> None:
    """加载插件、建 FP16 引擎；与 **fp32 eager** 参考对比（同一组 fp16 权重/输入，导出 ONNX 与推理共用）。"""

    plugin_so = _require_plugin_so()
    dev = _require_cuda()
    b, t, h, inter = 2, 5, 32, 48
    x, w_gu, w_d = _stable_fp16_weights(dev, b, t, h, inter, seed=10003)
    y_ref_fp32 = _reference_fp32(x, w_gu, w_d)

    with tempfile.TemporaryDirectory() as td:
        onnx_p = Path(td) / "m.onnx"
        _export_onnx_fp16(onnx_p, dev, x, w_gu, w_d)
        load_gemma_fused_gated_mlp_plugin(plugin_so)
        init_trt_plugins_after_load()
        diag: list[str] = []
        plan = build_gemma_fused_gated_mlp_engine_from_onnx(onnx_p, use_bf16=False, diagnostics_out=diag)
        if plan is None:
            body = "\n".join(diag) if diag else "(no extra diagnostics)"
            _emit_failure("TensorRT build_serialized_network failed or ONNX parse failed before plan", body)
            msg = "build_gemma_fused_gated_mlp_engine_from_onnx returned None (see stderr)"
            if _require_trt_plugin_strict():
                pytest.fail(msg)
            pytest.skip(msg)
        y_trt = run_gemma_fused_gated_mlp_engine(plan, x=x, gate_up_weight=w_gu, down_weight=w_d)

    assert y_trt.shape == (b, t, h)
    assert y_trt.dtype == torch.float16
    _assert_trt_matches_reference(y_trt, y_ref_fp32)
