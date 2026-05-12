# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""将 ``trt::GemmaFusedGatedMlp`` 的权重从 **图输入** 折叠为 **张量属性**，供 TensorRT 插件 v2 消除 ``Move``。"""

from __future__ import annotations

import copy
from typing import Iterable

import onnx
from onnx import numpy_helper


def embed_gemma_fused_gated_mlp_trt_static_weights(
    model: onnx.ModelProto,
    *,
    remove_embedded_initializers: bool = True,
    initializer_allowlist: Iterable[str] | None = None,
) -> onnx.ModelProto:
    """把 ``domain==trt``、``op_type==GemmaFusedGatedMlp`` 且 **3 输入** 的节点改为 **1 输入 + 权重张量属性**。

    折叠后 TensorRT 插件 **GemmaFusedGatedMlp** 版本 **"2"** 在建图时从 ``PluginField`` 读入权重并写入引擎序列化，
    推理期仅绑定 ``x`` / ``y``，从而避免两条大权重边上的 ``__myl_Move_*``。

    要求 ``gate_up_weight`` / ``down_weight`` 对应的输入名在 ``graph.initializer`` 中有 **独立** 常量，
    且每个 initializer **仅被该插件节点引用一次**（不与其它算子共享同一张量名）。

    Args:
        model: 已加载的 ONNX（会被 **原地** 修改；若需保留原图请先 ``copy.deepcopy``）。
        remove_embedded_initializers: 为真时，在折叠后从 ``graph.initializer`` 中删除已嵌入的权重，
            减小 ONNX 体积并避免无用常量。
        initializer_allowlist: 若给定，仅允许移除这些名字内的 initializer（安全闸）；
            默认 ``None`` 表示移除所有已嵌入且确认无引用的 initializer。

    Returns:
        传入的 ``model``（原地修改后的引用）。
    """

    graph = model.graph
    init_by_name = {t.name: t for t in graph.initializer}
    ref_count: dict[str, int] = {n: 0 for n in init_by_name}
    for n in graph.node:
        for x in n.input:
            if x in ref_count:
                ref_count[x] += 1

    allow = {str(s) for s in initializer_allowlist} if initializer_allowlist is not None else None
    to_remove: set[str] = set()

    for node in graph.node:
        if node.domain != "trt" or node.op_type != "GemmaFusedGatedMlp":
            continue
        if len(node.input) != 3:
            continue
        x_in, gu_in, d_in = node.input
        if gu_in not in init_by_name or d_in not in init_by_name:
            raise ValueError(
                f"GemmaFusedGatedMlp node {node.name!r}: gate_up_weight {gu_in!r} or down_weight {d_in!r} "
                "is not a graph initializer; static embed requires constant ONNX initializers."
            )
        if ref_count.get(gu_in, 0) != 1 or ref_count.get(d_in, 0) != 1:
            raise ValueError(
                f"Cannot safely embed weights for {node.name!r}: ref_count {gu_in}={ref_count.get(gu_in)}, "
                f"{d_in}={ref_count.get(d_in)} (expected 1 each; shared initializers need graph surgery first)."
            )

        gu_t = init_by_name[gu_in]
        d_t = init_by_name[d_in]
        gu = numpy_helper.to_array(gu_t)
        dd = numpy_helper.to_array(d_t)
        if gu.dtype != dd.dtype:
            raise ValueError(f"Mismatched weight dtypes: gate_up {gu.dtype}, down {dd.dtype}")
        if gu.ndim != 2 or dd.ndim != 2:
            raise ValueError(f"Bad weight ranks: gate_up {gu.shape}, down {dd.shape}")
        two_i, h_gu = int(gu.shape[0]), int(gu.shape[1])
        h_dn, inter = int(dd.shape[0]), int(dd.shape[1])
        if two_i % 2 != 0:
            raise ValueError(f"gate_up first dim must be even (2*I), got {two_i}")
        inter_g = two_i // 2
        if inter_g != inter or h_gu != h_dn:
            raise ValueError(
                f"Shape mismatch for fused MLP: gate_up {gu.shape}, down {dd.shape} "
                "(expect [2I,H] and [H,I] with same H,I)"
            )

        new_attr = [copy.deepcopy(a) for a in node.attribute]
        new_attr.append(onnx.helper.make_attribute("hidden_dim", h_gu))
        new_attr.append(onnx.helper.make_attribute("inter_dim", inter))
        new_attr.append(onnx.helper.make_attribute("plugin_version", 2))
        new_attr.append(onnx.helper.make_attribute("gate_up_weight", gu_t))
        new_attr.append(onnx.helper.make_attribute("down_weight", d_t))

        del node.input[:]
        node.input.append(x_in)
        del node.attribute[:]
        node.attribute.extend(new_attr)

        if remove_embedded_initializers:
            for nm in (gu_in, d_in):
                if allow is not None and nm not in allow:
                    raise ValueError(
                        f"Refusing to remove initializer {nm!r}: not in initializer_allowlist "
                        f"(set remove_embedded_initializers=False or widen allowlist)."
                    )
                to_remove.add(nm)

    if to_remove:
        graph.initializer[:] = [t for t in graph.initializer if t.name not in to_remove]

    return model
