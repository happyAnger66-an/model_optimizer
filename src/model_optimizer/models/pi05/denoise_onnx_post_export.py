# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Pi0.5 ``denoise.onnx`` 导出后的 TRT 强类型 dtype 修补。
#
# TensorRT 解析 ``MatMul``/``Gemm`` 与 bias 融合时要求 **主乘与 bias 元素类型一致**。ModelOpt
# 导出里常出现：QDQ 子图在 ONNX 中标 FLOAT、TRT 按 BF16 执行；或 NVFP4+FP8 混部导致 **投影层 /
# ``gemma_expert``** 上 FLOAT bias 与 BF16 主乘（或反之）不一致。
#
# ---------------------------------------------------------------------------
# 四步流水线（顺序固定，勿打乱）
# ---------------------------------------------------------------------------
#
# 1. **全局 FLOAT→BF16（bias initializer）**  
#    对 **非** ``_SKIP_GLOBAL_FLOAT_TO_BF16`` 子串的 bias：把仍为 FLOAT 的 bias 升为 BF16，与 BF16
#    MatMul 对齐。跳过 ``action_*``/``time_mlp_*``/``action_out_proj``（可能纯 FP32）及 ``gemma_expert``
#    （FP8 Float 累加时 bias 须保持 Float）。
#
# 2. **前缀投影 ``FLOAT→BF16``（图感知）**  
#    仅 ``action_in_proj``、``time_mlp_*``、``action_out_proj``：``Add(MatMul, bias)`` 与 ``Gemm`` 第三路
#    C=bias；当启发式判定线性主乘在 TRT 侧为 BF16 时，把 FLOAT bias 升为 BF16。
#
# 3. **`gemma_expert`` + NVFP4：``Gemm`` C 路 FLOAT→BF16**  
#    A/B 子串含 FP4 QDQ 且判定为 BF16 累加时，FLOAT bias → BF16（与步骤 1 跳过 gemma 互补）。
#
# 4. **`gemma_expert`` + FP8：逐 ``Gemm`` 选择性 BF16→FLOAT**  
#    仅当图中存在 FP8 DQ **且** 该 ``Gemm`` 为 Float 累加路径时，把 BF16 bias 降为 FLOAT；**不**处理
#    FP4+BF16 的 ``Gemm``（避免与步骤 3 冲突）。
#
# 入口：:func:`apply_denoise_onnx_post_export_patches`。

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# 全局 FLOAT→BF16 时跳过（见模块 docstring）。
_SKIP_GLOBAL_FLOAT_TO_BF16: tuple[str, ...] = (
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
    "gemma_expert",
)

# 步骤 2：仅这些前缀下的 bias 做图感知 FLOAT→BF16。
_PREFIX_PROJ_BIAS_SUBSTR: tuple[str, ...] = (
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
)


def _try_import_onnx():
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        return np, onnx, TensorProto, helper, numpy_helper
    except ImportError:
        return None


def _maybe_infer_shapes(model: Any) -> None:
    try:
        from onnx import shape_inference

        shape_inference.infer_shapes(model)
    except Exception:
        pass


def _onnx_tensor_elem_type(model: Any, tensor_name: str) -> int | None:
    from onnx import TensorProto

    for vi in model.graph.value_info:
        if vi.name == tensor_name and vi.type.HasField("tensor_type"):
            return int(vi.type.tensor_type.elem_type)
    for out in model.graph.output:
        if out.name == tensor_name and out.type.HasField("tensor_type"):
            return int(out.type.tensor_type.elem_type)
    for inp in model.graph.input:
        if inp.name == tensor_name and inp.type.HasField("tensor_type"):
            return int(inp.type.tensor_type.elem_type)
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return int(init.data_type)
    return None


def _onnx_producer_map(model: Any) -> dict[str, Any]:
    m: dict[str, Any] = {}
    for node in model.graph.node:
        for o in node.output:
            m[o] = node
    return m


def _onnx_trace_to_matmul_or_gemm(
    producers: dict[str, Any], tensor_name: str, *, max_hops: int = 8
) -> Any:
    cur = tensor_name
    for _ in range(max_hops):
        node = producers.get(cur)
        if node is None:
            return None
        if node.op_type in ("MatMul", "Gemm"):
            return node
        if not node.input:
            return None
        if len(node.input) >= 1 and node.op_type in (
            "Cast",
            "Transpose",
            "Reshape",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
        ):
            cur = node.input[0]
            continue
        return None
    return None


def _onnx_matmul_gemm_output_elem_type(model: Any, mm_node: Any, out_tensor: str) -> int | None:
    from onnx import TensorProto

    t = _onnx_tensor_elem_type(model, out_tensor)
    if t is not None and t != TensorProto.UNDEFINED:
        return t
    if mm_node.op_type == "MatMul" and len(mm_node.input) >= 2:
        t0 = _onnx_tensor_elem_type(model, mm_node.input[0])
        t1 = _onnx_tensor_elem_type(model, mm_node.input[1])
        if t0 == TensorProto.BFLOAT16 and t1 == TensorProto.BFLOAT16:
            return TensorProto.BFLOAT16
        if t0 == TensorProto.FLOAT and t1 == TensorProto.FLOAT:
            return TensorProto.FLOAT
    if mm_node.op_type == "Gemm" and len(mm_node.input) >= 2:
        t0 = _onnx_tensor_elem_type(model, mm_node.input[0])
        t1 = _onnx_tensor_elem_type(model, mm_node.input[1])
        if t0 == TensorProto.BFLOAT16 and t1 == TensorProto.BFLOAT16:
            return TensorProto.BFLOAT16
        if t0 == TensorProto.FLOAT and t1 == TensorProto.FLOAT:
            return TensorProto.FLOAT
    return None


def _onnx_prefix_linear_needs_bf16_bias(model: Any, lin_node: Any, out_tensor: str) -> bool:
    """MatMul/Gemm 的 A、B 在 TRT 侧是否按 BF16 与 bias 对齐（仅用 A/B 名 + 可选类型）。"""
    from onnx import TensorProto

    if lin_node.op_type == "MatMul":
        ins = list(lin_node.input)
    elif lin_node.op_type == "Gemm":
        ins = list(lin_node.input[:2])
    else:
        return False

    hay = " ".join(ins).lower()
    qdq_or_lowprec = any(
        k in hay
        for k in (
            "dequant",
            "quantize",
            "quantizer",
            "trt_fp8",
            "trt_fp4",
            "fp4qdq",
            "qdq",
            "nvfp4",
            "_f16",
            "bfloat16",
            "fp8",
            "mxfp",
            "float8",
        )
    )
    if qdq_or_lowprec:
        return True

    t = _onnx_matmul_gemm_output_elem_type(model, lin_node, out_tensor)
    if t == TensorProto.BFLOAT16:
        return True
    if t == TensorProto.FLOAT:
        return False
    return False


def _onnx_graph_has_fp8_dequantize(model: Any) -> bool:
    needle = "TRT_FP8DequantizeLinear"
    for n in model.graph.node:
        if n.op_type == needle or ("FP8" in n.op_type and "Dequant" in n.op_type):
            return True
        if any(needle in (x or "") for x in list(n.input) + list(n.output)):
            return True
    return False


def _onnx_hay_suggests_fp4_linear(hay: str) -> bool:
    h = hay.lower()
    return any(
        k in h
        for k in (
            "trt_fp4",
            "fp4qdq",
            "nvfp4",
            "mxfp4",
            "e2m1",
        )
    )


def _init_float_to_bf16_inplace(
    init: Any, np: Any, TensorProto: Any, helper: Any, numpy_helper: Any, base_dir: str
) -> bool:
    arr = numpy_helper.to_array(init, base_dir=base_dir)
    if arr.size == 0:
        return False
    arr_f = np.ascontiguousarray(arr, dtype=np.float32)
    t_bf = torch.from_numpy(arr_f).to(torch.bfloat16).contiguous()
    raw = t_bf.view(torch.uint16).detach().cpu().numpy().tobytes()
    new_tp = helper.make_tensor(
        init.name,
        TensorProto.BFLOAT16,
        [int(d) for d in arr_f.shape],
        raw,
        raw=True,
    )
    init.CopyFrom(new_tp)
    return True


def _step_global_float_bias_to_bf16(model: Any, base_dir: str) -> list[str]:
    imports = _try_import_onnx()
    if imports is None:
        return []
    np, onnx, TensorProto, helper, numpy_helper = imports  # noqa: F841

    changed: list[str] = []
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if not (init.name.endswith("bias") or init.name.endswith("/bias")):
            continue
        if any(s in init.name for s in _SKIP_GLOBAL_FLOAT_TO_BF16):
            continue
        try:
            arr = numpy_helper.to_array(init, base_dir=base_dir)
            if arr.size == 0:
                continue
            arr_f = np.ascontiguousarray(arr, dtype=np.float32)
            t_bf = torch.from_numpy(arr_f).to(torch.bfloat16).contiguous()
            raw = t_bf.view(torch.uint16).detach().cpu().numpy().tobytes()
            new_tp = helper.make_tensor(
                init.name,
                TensorProto.BFLOAT16,
                [int(d) for d in arr_f.shape],
                raw,
                raw=True,
            )
            init.CopyFrom(new_tp)
            changed.append(init.name)
        except Exception as exc:
            logger.warning("Skip global bias patch for %s: %s", init.name, exc)
    return changed


def _step_prefix_proj_float_bias_for_bf16_linear(model: Any, base_dir: str) -> list[str]:
    imports = _try_import_onnx()
    if imports is None:
        return []
    np, onnx, TensorProto, helper, numpy_helper = imports  # noqa: F841

    init_by_name = {init.name: init for init in model.graph.initializer}
    producers = _onnx_producer_map(model)
    changed: list[str] = []

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        a, b = node.input[0], node.input[1]
        if a in init_by_name and (a.endswith("bias") or a.endswith("/bias")):
            bias_name, other = a, b
        elif b in init_by_name and (b.endswith("bias") or b.endswith("/bias")):
            bias_name, other = b, a
        else:
            continue
        if not any(s in bias_name for s in _PREFIX_PROJ_BIAS_SUBSTR):
            continue
        bias_init = init_by_name.get(bias_name)
        if bias_init is None or bias_init.data_type != TensorProto.FLOAT:
            continue
        mm = _onnx_trace_to_matmul_or_gemm(producers, other)
        if mm is None or not _onnx_prefix_linear_needs_bf16_bias(model, mm, other):
            continue
        try:
            if _init_float_to_bf16_inplace(
                bias_init, np, TensorProto, helper, numpy_helper, base_dir
            ):
                changed.append(bias_name)
        except Exception as exc:
            logger.warning("Skip prefix bias patch for %s: %s", bias_name, exc)

    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 3:
            continue
        bias_name = node.input[2]
        bias_init = init_by_name.get(bias_name)
        if bias_init is None or bias_init.data_type != TensorProto.FLOAT:
            continue
        if not (bias_name.endswith("bias") or bias_name.endswith("/bias")):
            continue
        if not any(s in bias_name for s in _PREFIX_PROJ_BIAS_SUBSTR):
            continue
        out_tensor = node.output[0] if node.output else ""
        if not _onnx_prefix_linear_needs_bf16_bias(model, node, out_tensor):
            continue
        try:
            if _init_float_to_bf16_inplace(
                bias_init, np, TensorProto, helper, numpy_helper, base_dir
            ):
                if bias_name not in changed:
                    changed.append(bias_name)
        except Exception as exc:
            logger.warning("Skip prefix Gemm bias patch for %s: %s", bias_name, exc)

    return changed


def _step_gemma_fp4_float_bias_to_bf16_gemm(model: Any, base_dir: str) -> list[str]:
    imports = _try_import_onnx()
    if imports is None:
        return []
    np, onnx, TensorProto, helper, numpy_helper = imports  # noqa: F841

    init_by_name = {init.name: init for init in model.graph.initializer}
    changed: list[str] = []

    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 3:
            continue
        bias_name = node.input[2]
        bias_init = init_by_name.get(bias_name)
        if bias_init is None or bias_init.data_type != TensorProto.FLOAT:
            continue
        if "gemma_expert" not in bias_name:
            continue
        if not (bias_name.endswith("bias") or bias_name.endswith("/bias")):
            continue
        hay = " ".join(node.input[:2]).lower()
        if not _onnx_hay_suggests_fp4_linear(hay):
            continue
        out_tensor = node.output[0] if node.output else ""
        if not _onnx_prefix_linear_needs_bf16_bias(model, node, out_tensor):
            continue
        try:
            if _init_float_to_bf16_inplace(
                bias_init, np, TensorProto, helper, numpy_helper, base_dir
            ):
                changed.append(bias_name)
        except Exception as exc:
            logger.warning("Skip gemma FP4 bias patch for %s: %s", bias_name, exc)
    return changed


def _step_gemma_fp8_bf16_bias_to_float_gemm(model: Any, base_dir: str) -> list[str]:
    imports = _try_import_onnx()
    if imports is None:
        return []
    np, onnx, TensorProto, numpy_helper = imports

    if not _onnx_graph_has_fp8_dequantize(model):
        return []

    init_by_name = {init.name: init for init in model.graph.initializer}
    bias_to_float: set[str] = set()

    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 3:
            continue
        bias_name = node.input[2]
        init = init_by_name.get(bias_name)
        if init is None or init.data_type != TensorProto.BFLOAT16:
            continue
        if "gemma_expert" not in bias_name:
            continue
        if not (bias_name.endswith("bias") or bias_name.endswith("/bias")):
            continue

        hay = " ".join(node.input[:2]).lower()
        out_tensor = node.output[0] if node.output else ""
        t0 = _onnx_tensor_elem_type(model, node.input[0])
        t1 = _onnx_tensor_elem_type(model, node.input[1])

        if t0 == TensorProto.BFLOAT16 or t1 == TensorProto.BFLOAT16:
            continue
        if _onnx_hay_suggests_fp4_linear(hay) and _onnx_prefix_linear_needs_bf16_bias(
            model, node, out_tensor
        ):
            continue
        bias_to_float.add(bias_name)

    changed: list[str] = []
    for init in model.graph.initializer:
        if init.name not in bias_to_float:
            continue
        try:
            arr = numpy_helper.to_array(init, base_dir=base_dir)
            arr_f = np.ascontiguousarray(arr, dtype=np.float32)
            new_tp = numpy_helper.from_array(arr_f, name=init.name)
            init.CopyFrom(new_tp)
            changed.append(init.name)
        except Exception as exc:
            logger.warning("Skip gemma BF16->float for %s: %s", init.name, exc)
    return changed


def apply_denoise_onnx_post_export_patches(onnx_path: str) -> None:
    """对 ``denoise.onnx`` 依次应用四步 TRT dtype 修补（见模块顶部说明）。"""
    imports = _try_import_onnx()
    if imports is None:
        logger.warning("onnx/numpy unavailable; skip denoise ONNX post-export patches for %s", onnx_path)
        return
    _, onnx, _, _, _ = imports

    if not os.path.isfile(onnx_path):
        return

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        logger.warning("Failed to load ONNX for post-export patches %s: %s", onnx_path, exc)
        return

    _maybe_infer_shapes(model)
    base_dir = os.path.dirname(os.path.abspath(onnx_path))

    log_parts: list[str] = []
    c1 = _step_global_float_bias_to_bf16(model, base_dir)
    if c1:
        log_parts.append(f"global_bf16_bias={len(c1)}")
    c2 = _step_prefix_proj_float_bias_for_bf16_linear(model, base_dir)
    if c2:
        log_parts.append(f"prefix_bf16_bias={len(c2)}")
    c3 = _step_gemma_fp4_float_bias_to_bf16_gemm(model, base_dir)
    if c3:
        log_parts.append(f"gemma_fp4_bf16_bias={len(c3)}")
    c4 = _step_gemma_fp8_bf16_bias_to_float_gemm(model, base_dir)
    if c4:
        log_parts.append(f"gemma_fp8_float_bias={len(c4)}")

    if not (c1 or c2 or c3 or c4):
        return

    try:
        onnx.save(model, onnx_path)
    except Exception as exc:
        logger.warning("Failed to save ONNX after post-export patches %s: %s", onnx_path, exc)
        return

    logger.info(
        "denoise ONNX post-export patches applied %s: %s",
        onnx_path,
        ", ".join(log_parts) if log_parts else "(ok)",
    )
