# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from model_optimizer.ops.siglip_mlp import SiglipMlpCustomOpWrapper, siglip_mlp_eager


class _FakeSiglipMlp(nn.Module):
    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="gelu")
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


def test_siglip_mlp_eager_matches_reference_mlp() -> None:
    torch.manual_seed(0)
    m = _FakeSiglipMlp(8, 16)
    x = torch.randn(2, 5, 8)
    y_ref = m(x)
    y_fus = siglip_mlp_eager(
        x, m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias, act_id=0
    )
    assert torch.allclose(y_ref, y_fus, rtol=1e-5, atol=1e-6)


def test_wrapper_matches_reference_mlp() -> None:
    torch.manual_seed(1)
    base = _FakeSiglipMlp(12, 24)
    wrap = SiglipMlpCustomOpWrapper(base)
    x = torch.randn(3, 7, 12)
    assert torch.allclose(base(x), wrap(x), rtol=1e-5, atol=1e-6)
