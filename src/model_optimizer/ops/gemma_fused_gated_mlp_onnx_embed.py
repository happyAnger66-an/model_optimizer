# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""将 ``trt::GemmaFusedGatedMlp`` 的权重从 **图输入** 折叠为 **张量属性**，供 TensorRT 插件 v2 消除 ``Move``。"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper


def embed_gemma_fused_gated_mlp_trt_static_weights(
    model: onnx.ModelProto,
    *,
    remove_embedded_initializers: bool = True,
    initializer_allowlist: Iterable[str] | None = None,
    bake_weights_as_float16: bool = False,
) -> onnx.ModelProto:
    """把 ``domain==trt``、``op_type==GemmaFusedGatedMlp`` 且 **3 输入** 的节点改为 **1 输入 + 权重张量属性**。

    折叠后 TensorRT 插件 **GemmaFusedGatedMlp** 版本 **"2"** 在建图时从 ``PluginField`` 读入权重并写入引擎序列化，
    推理期仅绑定 ``x`` / ``y``，从而避免两条大权重边上的 ``__myl_Move_*``。

    要求 ``gate_up_weight`` / ``down_weight`` 对应的输入名在 ``graph.initializer`` 中有 **独立** 常量，
    且每个 initializer **仅被该插件节点引用一次**（不与其它算子共享同一张量名）。

    **FP16 引擎与 dtype**：插件 v2 若从 ONNX 读到 **fp32** 权重，会在 TRT 侧 **量化为 bf16 字节**
    （``io_type=1``）；此时激活多为 **fp16**，会在 ``enqueue`` 报错。若 initializer 已是 **bf16** 而
    主链为 **fp16**，同样不一致。对 FP16 建引擎 / fp16 激活，请令嵌入属性为 **FLOAT16**：
    导出时即用 fp16 权重，或在本函数设 ``bake_weights_as_float16=True``（先转 ``float32`` 再 ``float16``）。

    Args:
        model: 已加载的 ONNX（会被 **原地** 修改；若需保留原图请先 ``copy.deepcopy``）。
        remove_embedded_initializers: 为真时，在折叠后从 ``graph.initializer`` 中删除已嵌入的权重，
            减小 ONNX 体积并避免无用常量。
        initializer_allowlist: 若给定，仅允许移除这些名字内的 initializer（安全闸）；
            默认 ``None`` 表示移除所有已嵌入且确认无引用的 initializer。
        bake_weights_as_float16: 为真时，在写入节点属性前将两份权重转为 **ONNX FLOAT16**
            ``TensorProto``（与 ``BuilderFlag.FP16`` 下插件输入 ``x`` 一致）。

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

        gu_attr = gu_t
        d_attr = d_t
        if bake_weights_as_float16:
            gu_f32 = np.asarray(gu, dtype=np.float32)
            dd_f32 = np.asarray(dd, dtype=np.float32)
            gu_attr = numpy_helper.from_array(gu_f32.astype(np.float16, copy=False))
            d_attr = numpy_helper.from_array(dd_f32.astype(np.float16, copy=False))

        new_attr = [copy.deepcopy(a) for a in node.attribute]
        new_attr.append(onnx.helper.make_attribute("hidden_dim", h_gu))
        new_attr.append(onnx.helper.make_attribute("inter_dim", inter))
        # ONNX-TensorRT FallbackPluginImporter 用 **字符串** 属性 ``plugin_version`` / ``plugin_namespace``
        # 调 ``IPluginRegistry::getCreator``（见 onnx-tensorrt onnxOpCheckers.cpp）；写 INT 会导致版本回退/查不到。
        new_attr.append(onnx.helper.make_attribute("plugin_version", "2"))
        new_attr.append(onnx.helper.make_attribute("plugin_namespace", "trt"))
        new_attr.append(onnx.helper.make_attribute("gate_up_weight", gu_attr))
        new_attr.append(onnx.helper.make_attribute("down_weight", d_attr))

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
        # Protobuf repeated fields reject `initializer[:] = [...]` on some versions.
        for i in reversed(range(len(graph.initializer))):
            if graph.initializer[i].name in to_remove:
                del graph.initializer[i]

    return model
