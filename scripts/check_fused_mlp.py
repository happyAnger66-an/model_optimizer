# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""快速校验 ``fused_mlp`` 特性在 ``pi05_libero/llm_with_cutedsl`` 路径下是否生效。

用法::

    python scripts/check_fused_mlp.py \
        --model_path /srcs/openpi/pytorch_pi05_libero/ \
        --feature_config config/export_llm.json

分别报告：

1. 每层 ``layer.mlp`` 类型（``FusedGemmaMLP`` 与否）。
2. 是否存在合并后的 ``gate_up_proj`` 以及其权重形状（应为 ``[2I, H]``）。
3. 与原始权重的数学等价性验证（随机激活 ``x``，对比 ``FusedGemmaMLP(x)``
   与 ``down(act(gate(x)) * up(x))`` 的最大绝对误差）。
"""

from __future__ import annotations

import argparse

import torch

from model_optimizer.config.feature_config import FeatureConfig
from model_optimizer.models.pi05.fused_mlp import FusedGemmaMLP
from model_optimizer.models.pi05.llm_with_cutedsl import LLMWithCuteDsl
from model_optimizer.models.pi05.model_pi05 import Pi05Model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="pi05_libero")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--feature_config", default=None)
    parser.add_argument("--train_config", default=None)
    parser.add_argument("--n_samples", type=int, default=2)
    args = parser.parse_args()

    fc = FeatureConfig.load(args.feature_config)
    print(f"[check] feature_config: {fc}")

    pi05 = Pi05Model.construct_from_name_path(args.model_name, args.model_path, args.train_config)
    model = LLMWithCuteDsl.construct_model(pi05, feature_config=fc)
    gemma = model.model._hf_gemma

    # 1) 每层 mlp 类型
    fused_layers = []
    unfused_layers = []
    for idx, layer in enumerate(gemma.layers):
        mlp = layer.mlp
        kind = type(mlp).__name__
        (fused_layers if isinstance(mlp, FusedGemmaMLP) else unfused_layers).append((idx, kind))

    print(f"[check] total layers: {len(gemma.layers)}")
    print(f"[check] fused layers : {len(fused_layers)} -> {fused_layers[:3]}{'...' if len(fused_layers) > 3 else ''}")
    print(f"[check] unfused      : {len(unfused_layers)} -> {unfused_layers[:3]}{'...' if len(unfused_layers) > 3 else ''}")

    if not fused_layers:
        print("[check] FAIL: no layer fused; fused_mlp likely OFF.")
        return 1

    # 2) gate_up_proj 形状
    sample_idx = fused_layers[0][0]
    sample_mlp: FusedGemmaMLP = gemma.layers[sample_idx].mlp
    H = sample_mlp.hidden_size
    I = sample_mlp.intermediate_size
    w = sample_mlp.gate_up_proj.weight
    print(f"[check] gate_up_proj.weight shape={tuple(w.shape)} (expect [2I={2 * I}, H={H}])")
    assert tuple(w.shape) == (2 * I, H), f"unexpected gate_up_proj shape {w.shape}"

    # 3) 数学等价性：FusedGemmaMLP(x) == down(act(gate(x)) * up(x))
    device = next(sample_mlp.parameters()).device
    dtype = next(sample_mlp.parameters()).dtype
    x = torch.randn(args.n_samples, 8, H, device=device, dtype=dtype)
    with torch.inference_mode():
        y_fused = sample_mlp(x)

        gate_w = w[:I, :]
        up_w = w[I:, :]
        g = torch.nn.functional.linear(x, gate_w)
        u = torch.nn.functional.linear(x, up_w)
        h = sample_mlp.act_fn(g) * u
        y_ref = sample_mlp.down_proj(h)

    err = (y_fused - y_ref).abs().max().item()
    print(f"[check] max |fused - reference| = {err:.3e} (dtype={dtype})")
    tol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
    if err > tol:
        print(f"[check] FAIL: numerical mismatch > tol={tol}")
        return 2

    print("[check] OK: fused_mlp 已生效，权重形状与数学等价均通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
