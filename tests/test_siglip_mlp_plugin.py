# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
import tempfile

import onnx
import torch
import torch.nn as nn
from model_optimizer.ops.siglip_mlp import siglip_mlp_eager
from model_optimizer.ops.siglip_mlp_plugin import (
    register_siglip_mlp_plugin_onnx_symbolic_functions,
    siglip_mlp_plugin,
)


def test_siglip_mlp_plugin_matches_eager() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    w1, b1 = torch.randn(16, 8), torch.randn(16)
    w2, b2 = torch.randn(8, 16), torch.randn(8)
    y0 = siglip_mlp_eager(x, w1, b1, w2, b2, 0)
    y1 = siglip_mlp_plugin(x, w1, b1, w2, b2, 0)
    assert torch.allclose(y0, y1, rtol=1e-5, atol=1e-6)


def test_onnx_export_contains_siglip_mlp_plugin_node() -> None:
    register_siglip_mlp_plugin_onnx_symbolic_functions()

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(16, 8))
            self.b1 = nn.Parameter(torch.randn(16))
            self.w2 = nn.Parameter(torch.randn(8, 16))
            self.b2 = nn.Parameter(torch.randn(8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return siglip_mlp_plugin(x, self.w1, self.b1, self.w2, self.b2, 0)

    m = Toy().eval()
    x = torch.randn(1, 4, 8)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
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
        model = onnx.load(path)
    finally:
        os.unlink(path)

    plugin_nodes = [n for n in model.graph.node if n.op_type == "SiglipMlpPlugin"]
    assert plugin_nodes, f"got op_types={[n.op_type for n in model.graph.node]}"
    assert plugin_nodes[0].domain == "trt"
