# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
import torch.nn as nn


def test_gemma_onnx_gqa_attention_function_emits_attention_op(tmp_path):
    from model_optimizer.ops.gemma_onnx_attention_export import GemmaOnnxGqaAttentionFunction

    class Mod(nn.Module):
        def forward(self, q, k, v, mask, scaling: float, dropout_p: float):
            return GemmaOnnxGqaAttentionFunction.apply(q, k, v, mask, scaling, dropout_p)

    b, hq, hkv, s, d = 1, 8, 1, 4, 16
    q = torch.randn(b, hq, s, d)
    k = torch.randn(b, hkv, s, d)
    v = torch.randn(b, hkv, s, d)
    mask = torch.zeros(b, 1, s, s)
    scaling = float(d**-0.5)
    out = Mod().eval()(q, k, v, mask, scaling, 0.0)
    assert tuple(out.shape) == (b, s, hq, d)

    onnx_path = tmp_path / "attn.onnx"
    torch.onnx.export(
        Mod().eval(),
        (q, k, v, mask, scaling, 0.0),
        str(onnx_path),
        input_names=["q", "k", "v", "mask", "scaling", "dropout_p"],
        output_names=["y"],
        opset_version=int(os.environ.get("MODEL_OPTIMIZER_GEMMA_ONNX_OPSET", "23")),
        dynamo=False,
    )
    import onnx

    types = [n.op_type for n in onnx.load(str(onnx_path)).graph.node]
    assert "Attention" in types
    assert "Transpose" in types
