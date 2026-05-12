# Copyright 2026 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fused Gemma MLP (GeGLU) and ONNX custom op."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from model_optimizer.ops.gemma_fused_gated_mlp import (
    FusedGemmaMLP,
    gemma_fused_gated_mlp_eager,
    patch_decoder_fused_gated_mlp,
)
from model_optimizer.ops.gemma_fused_gated_mlp_plugin import (
    gemma_fused_gated_mlp_plugin,
    register_gemma_fused_gated_mlp_onnx_symbolic_functions,
)


class _TinyCfg:
    hidden_size = 32
    intermediate_size = 48
    hidden_act = "gelu_pytorch_tanh"


class _RefGemmaMLP(nn.Module):
    """Same math as OpenPI ``GemmaMLP`` without importing full transformers stack."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = torch.nn.functional.gelu(g, approximate="tanh") * u
        return self.down_proj(h)


def test_fused_matches_reference_bf16() -> None:
    torch.manual_seed(0)
    cfg = _TinyCfg()
    ref = _RefGemmaMLP(cfg).to(dtype=torch.bfloat16)
    fused = FusedGemmaMLP.from_gemma_mlp(ref).to(dtype=torch.bfloat16)
    x = torch.randn(2, 7, cfg.hidden_size, dtype=torch.bfloat16)
    y0 = ref(x)
    y1 = fused(x)
    assert torch.allclose(y0, y1, rtol=2e-2, atol=2e-2)


def test_eager_equals_plugin_act_id_1() -> None:
    torch.manual_seed(1)
    h, i = 16, 24
    x = torch.randn(3, 5, h, dtype=torch.float16)
    w_gu = torch.randn(2 * i, h, dtype=torch.float16)
    w_d = torch.randn(h, i, dtype=torch.float16)
    y0 = gemma_fused_gated_mlp_eager(x, w_gu, w_d, 1)
    y1 = gemma_fused_gated_mlp_plugin(x, w_gu, w_d, 1)
    assert torch.equal(y0, y1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_onnx_export_contains_fused_node() -> None:
    register_gemma_fused_gated_mlp_onnx_symbolic_functions()
    cfg = _TinyCfg()

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fused = FusedGemmaMLP(cfg)
            # avoid TRT export path inside FusedGemmaMLP.forward — call plugin directly
            self.w_gu = nn.Parameter(torch.randn(2 * cfg.intermediate_size, cfg.hidden_size, dtype=torch.float16))
            self.w_d = nn.Parameter(torch.randn(cfg.hidden_size, cfg.intermediate_size, dtype=torch.float16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return gemma_fused_gated_mlp_plugin(x, self.w_gu, self.w_d, 1)

    m = _Wrap().cuda().eval()
    x = torch.randn(1, 4, cfg.hidden_size, dtype=torch.float16, device="cuda")
    path = "/tmp/test_gemma_fused_mlp.onnx"
    try:
        torch.onnx.export(
            m,
            (x,),
            path,
            input_names=["x"],
            output_names=["y"],
            opset_version=19,
            dynamo=False,
        )
        import onnx

        model = onnx.load(path)
        nodes = [n for n in model.graph.node if n.op_type == "GemmaFusedGatedMlp"]
        assert nodes, f"expected GemmaFusedGatedMlp node, got {[n.op_type for n in model.graph.node]}"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_patch_decoder_skips_non_gelu_pytorch_tanh() -> None:
    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Identity()

    class _Dec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    d = _Dec()
    n = patch_decoder_fused_gated_mlp(d, enabled=True)
    assert n == 0
