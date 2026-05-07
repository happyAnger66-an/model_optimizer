"""π₀.₅ 单次 flow / denoise 步：对齐 openpi PI0Pytorch.denoise_step + embed_suffix（pi05 分支）。"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from transformers.cache_utils import DynamicCache

from ..model import Model
from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.quantization.quantization_utils import (
    quant_config_targets_hf_bmm_kv,
    quantize_model,
)
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

logger = logging.getLogger(__name__)


# 这些路径下 ONNX 上 **线性/Gemm 主结果** 常为 Float32，若把 FLOAT bias 强行改为 BF16，会在
# ``Add(Gemm/MatMul, bias)`` 触发 TRT ``ElementWiseOperation SUM must have same input types``：
# - ``action_*`` / ``time_mlp_*``：在 ``suffix_embs.to(expert_dtype)`` 之前与 ``x_t`` 一起走 FP32。
# - ``gemma_expert``：FP8 量化导出时 ``TRT_FP8DequantizeLinear`` 后输入 Gemm 为 Float，bias 必须保持
#   Float；否则会报 ``Gemm_output`` Float 与 broadcast bias BF16 不一致（如 ``input_layernorm/dense``）。
_SKIP_FLOAT_BIAS_TO_BF16_SUBSTR: tuple[str, ...] = (
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
    "gemma_expert",
)

# 与 ``_SKIP_FLOAT_BIAS_TO_BF16_SUBSTR`` 中投影子串对应；量化导出后 MatMul 常为 BF16，bias 仍为 FLOAT，
# 需按图单独把 FLOAT bias 升为 BF16（见 :func:`_patch_denoise_onnx_prefix_proj_float_bias_for_bf16_matmul_add`）。
_PREFIX_PROJ_BIAS_SUBSTR: tuple[str, ...] = (
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
)


def _patch_denoise_onnx_float_bias_to_bfloat16(onnx_path: str) -> None:
    """将仍为 FLOAT 的 Linear bias 初始值改为 BFLOAT16（仅针对 MatMul 已为 BF16 的分支）。

    TensorRT 强类型 / bf16 构建下，**部分**子图 MatMul 已为 BF16 而 ONNX 仍导出 FLOAT bias 时，
    统一 bias 为 BF16 与 MatMul 输出对齐。

    **不**转换 :data:`_SKIP_FLOAT_BIAS_TO_BF16_SUBSTR` 下的 bias（见模块级注释）：含 FP32 投影与
    FP8-DQ 后 Float Gemm 的 ``gemma_expert`` 全树。
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        logger.warning("onnx/numpy unavailable; skip bias dtype patch for %s", onnx_path)
        return

    if not os.path.isfile(onnx_path):
        return

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        logger.warning("Failed to load ONNX for bias patch %s: %s", onnx_path, exc)
        return

    base_dir = os.path.dirname(os.path.abspath(onnx_path))
    changed: list[str] = []

    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if not (init.name.endswith("bias") or init.name.endswith("/bias")):
            continue
        if any(s in init.name for s in _SKIP_FLOAT_BIAS_TO_BF16_SUBSTR):
            continue
        try:
            arr = numpy_helper.to_array(init, base_dir=base_dir)
        except Exception:
            continue
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

    if not changed:
        return

    try:
        onnx.save(model, onnx_path)
    except Exception as exc:
        logger.warning("Failed to save ONNX after bias patch %s: %s", onnx_path, exc)
        return

    preview = changed[:12]
    if len(changed) > 12:
        preview.append("...")
    logger.info(
        "Patched %d FLOAT->BFLOAT16 bias initializers in %s: %s",
        len(changed),
        onnx_path,
        preview,
    )


def _onnx_tensor_elem_type(model, tensor_name: str) -> int | None:
    """返回 ONNX 图中张量的 ``TensorProto`` elem_type，未知则 ``None``。"""
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


def _onnx_producer_map(model):
    """tensor 名 -> 产出该 tensor 的 node（假定单生产者）。"""
    m: dict[str, object] = {}
    for node in model.graph.node:
        for o in node.output:
            m[o] = node
    return m


def _onnx_trace_to_matmul_or_gemm(producers: dict, tensor_name: str, *, max_hops: int = 8):
    """沿单输入链回退，直到 ``MatMul`` / ``Gemm`` 或无法继续。"""
    cur = tensor_name
    for _ in range(max_hops):
        node = producers.get(cur)
        if node is None:
            return None
        if node.op_type in ("MatMul", "Gemm"):
            return node
        if not node.input:
            return None
        # Cast / Transpose / Reshape / Squeeze 等常见一元前驱
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


def _onnx_matmul_gemm_output_elem_type(model, mm_node, out_tensor: str) -> int | None:
    """推断 ``MatMul``/``Gemm`` 主输出 ``out_tensor`` 的元素类型。"""
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


def _onnx_prefix_linear_needs_bf16_bias(model, lin_node, out_tensor: str) -> bool:
    """判断 ``MatMul`` / ``Gemm``（仅 A、B 两路）在 TRT 中是否按 BF16 计算，从而 bias 需为 BF16。

    ModelOpt / ONNX 导出里 ``DequantizeLinear`` 等输出在图中常标为 **FLOAT**，但 TRT 仍按 BF16
    执行；仅靠 ``value_info`` 会漏判，故结合 **A/B 输入张量名** 启发式（不含 Gemm 第三路 C=bias）。
    """
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
    # 类型缺失且无名线索：保守不升 bias（避免纯 FP32 导出误伤）
    return False


def _patch_denoise_onnx_prefix_proj_float_bias_for_bf16_matmul_add(onnx_path: str) -> None:
    """``action_*`` / ``time_mlp_*`` 上 FLOAT bias 与 BF16 线性输出对齐。

    - ``Add(MatMul, bias_initializer)``（如 ``action_in_proj``）
    - ``Gemm`` 第三路 **C=bias**（如 ``time_mlp_in`` / ``time_mlp_out`` 导出为 Gemm 融合 bias）

    与 :func:`_patch_denoise_onnx_float_bias_to_bfloat16` 互补：投影层被排除在全局 FLOAT→BF16 之外；
    量化/NVFP4 路径下 TRT 按 BF16 做矩阵乘时，bias 必须为 BF16，否则 ``Gemm`` / ``Add`` 强类型失败。
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        logger.warning("onnx/numpy unavailable; skip prefix bias patch for %s", onnx_path)
        return

    if not os.path.isfile(onnx_path):
        return

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        logger.warning("Failed to load ONNX for prefix bias patch %s: %s", onnx_path, exc)
        return

    try:
        from onnx import shape_inference

        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    init_by_name = {init.name: init for init in model.graph.initializer}
    producers = _onnx_producer_map(model)
    changed: list[str] = []

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        a, b = node.input[0], node.input[1]
        bias_name: str | None = None
        other: str | None = None
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
        if mm is None:
            continue
        if not _onnx_prefix_linear_needs_bf16_bias(model, mm, other):
            continue
        try:
            arr = numpy_helper.to_array(
                bias_init,
                base_dir=os.path.dirname(os.path.abspath(onnx_path)),
            )
            if arr.size == 0:
                continue
            arr_f = np.ascontiguousarray(arr, dtype=np.float32)
            t_bf = torch.from_numpy(arr_f).to(torch.bfloat16).contiguous()
            raw = t_bf.view(torch.uint16).detach().cpu().numpy().tobytes()
            new_tp = helper.make_tensor(
                bias_name,
                TensorProto.BFLOAT16,
                [int(d) for d in arr_f.shape],
                raw,
                raw=True,
            )
            bias_init.CopyFrom(new_tp)
            changed.append(bias_name)
        except Exception as exc:
            logger.warning("Skip prefix bias patch for %s: %s", bias_name, exc)

    # ``Gemm`` 融合 bias 为第三输入 C，无单独 ``Add``（如 ``time_mlp_in`` / ``time_mlp_out``）。
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
            arr = numpy_helper.to_array(
                bias_init,
                base_dir=os.path.dirname(os.path.abspath(onnx_path)),
            )
            if arr.size == 0:
                continue
            arr_f = np.ascontiguousarray(arr, dtype=np.float32)
            t_bf = torch.from_numpy(arr_f).to(torch.bfloat16).contiguous()
            raw = t_bf.view(torch.uint16).detach().cpu().numpy().tobytes()
            new_tp = helper.make_tensor(
                bias_name,
                TensorProto.BFLOAT16,
                [int(d) for d in arr_f.shape],
                raw,
                raw=True,
            )
            bias_init.CopyFrom(new_tp)
            if bias_name not in changed:
                changed.append(bias_name)
        except Exception as exc:
            logger.warning("Skip prefix Gemm bias patch for %s: %s", bias_name, exc)

    if not changed:
        return

    try:
        onnx.save(model, onnx_path)
    except Exception as exc:
        logger.warning("Failed to save ONNX after prefix bias patch %s: %s", onnx_path, exc)
        return

    logger.info(
        "Patched %d prefix FLOAT->BFLOAT16 bias(es) for BF16 MatMul+Add / Gemm in %s: %s",
        len(changed),
        onnx_path,
        changed[:12],
    )


def _onnx_graph_has_fp8_dequantize(model) -> bool:
    """图中是否含 ModelOpt / TRT FP8 反量化节点（DQ 后算子多为 Float）。"""
    needle = "TRT_FP8DequantizeLinear"
    for n in model.graph.node:
        if n.op_type == needle or (
            "FP8" in n.op_type and "Dequant" in n.op_type
        ):
            return True
        if any(needle in (x or "") for x in list(n.input) + list(n.output)):
            return True
    return False


def _patch_denoise_onnx_gemma_bf16_bias_to_float_if_fp8(onnx_path: str) -> None:
    """FP8 导出图中 Gemm 输入经 DQ 为 Float 时，将 ``gemma_expert`` 下仍为 BF16 的 bias 改为 Float。

    否则 TRT 解析 ``Gemm``（Float 累加结果 + BF16 bias 广播）会在融合 ``Add`` 上报
    ``ElementWiseOperation SUM must have same input types``。
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, numpy_helper
    except ImportError:
        logger.warning("onnx/numpy unavailable; skip gemma BF16 bias patch for %s", onnx_path)
        return

    if not os.path.isfile(onnx_path):
        return

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        logger.warning("Failed to load ONNX for gemma bias patch %s: %s", onnx_path, exc)
        return

    if not _onnx_graph_has_fp8_dequantize(model):
        return

    base_dir = os.path.dirname(os.path.abspath(onnx_path))
    changed: list[str] = []

    for init in model.graph.initializer:
        if "gemma_expert" not in init.name:
            continue
        if not (init.name.endswith("bias") or init.name.endswith("/bias")):
            continue
        if init.data_type != TensorProto.BFLOAT16:
            continue
        try:
            arr = numpy_helper.to_array(init, base_dir=base_dir)
            arr_f = np.ascontiguousarray(arr, dtype=np.float32)
            new_tp = numpy_helper.from_array(arr_f, name=init.name)
        except Exception as exc:
            logger.warning("Skip gemma bias patch for %s: %s", init.name, exc)
            continue
        init.CopyFrom(new_tp)
        changed.append(init.name)

    if not changed:
        return

    try:
        onnx.save(model, onnx_path)
    except Exception as exc:
        logger.warning("Failed to save ONNX after gemma BF16 bias patch %s: %s", onnx_path, exc)
        return

    preview = changed[:12]
    if len(changed) > 12:
        preview.append("...")
    logger.info(
        "Patched %d BFLOAT16->FLOAT gemma_expert bias initializers (FP8 graph) in %s: %s",
        len(changed),
        onnx_path,
        preview,
    )


# 与 openpi pi0_pytorch 中 _prepare_attention_masks_4d 一致
_ATTN_MASK_FILL_VALUE = -2.3819763e38


def _get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """标量时间步 → sin-cos 向量（与 openpi pi0_pytorch 一致）。time 形状为 (batch_size,)。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dev = device if device is not None else time.device
    dtype = _get_safe_dtype(torch.float64, dev.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=dev)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """与 openpi pi0_pytorch.make_att_2d_masks 一致。"""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class Pi05DenoiseStep(nn.Module, Model):
    """
    单次去噪步：给定 prefix KV 缓存与当前噪声动作 x_t、时间 t，预测速度场 v_t。

    对应 openpi 中 ``denoise_step``（suffix 仅含 action tokens，pi05 不使用 state token）。
    Euler 更新 ``x_t += dt * v_t`` 应在图外由宿主循环调用。
    """

    def __init__(
        self,
        gemma_expert: nn.Module,
        expert_config,
        action_in_proj: nn.Linear,
        time_mlp_in: nn.Linear,
        time_mlp_out: nn.Linear,
        action_out_proj: nn.Linear,
        *,
        action_horizon: int,
        action_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gemma_expert = gemma_expert
        self.expert_config = expert_config
        self.action_in_proj = action_in_proj
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out
        self.action_out_proj = action_out_proj
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = gemma_expert.device

        self.gemma_expert.config._attn_implementation = "eager"  # noqa: SLF001

        suffix_ar = [1] + [0] * (action_horizon - 1)
        self.register_buffer(
            "_suffix_ar_mask",
            torch.tensor(suffix_ar, dtype=torch.int32),
            persistent=False,
        )

    @property
    def model(self):
        """与 Expert 一致，供 Model 基类 NVFP4 等路径使用。"""
        return self.gemma_expert

    def get_calibrate_dataset(self, calib_data):
        # 与 LLM/Vit/Expert 一致：支持 manifest+shards（低内存）与旧 merged .pt
        return open_pi05_calib_for_quantize(calib_data, component="pi05_denoise")

    def val(self, val_data, batch_size, output_dir):
        raise NotImplementedError(
            "Pi05DenoiseStep.val 未实现：需提供 denoise 步校准/对比数据与指标。"
        )

    def export(self, export_dir, export_dtype=torch.bfloat16, dynamo=False, mode=None):
        """导出单次 denoise 步 ONNX，输入输出与 ``forward`` 一致。

        ``past_keys`` / ``past_values`` 与 LLM 导出一致：``torch.cat(..., dim=0)`` 后为
        **4D** ``[num_layers, batch, prefix_seq_len, head_dim]``（head_dim 固定，勿标成动态轴）。

        默认 ``dynamo=False`` 使用传统导出器 + ``dynamic_axes``；若 ``dynamo=True``，须使用
        ``dynamic_shapes``（不再与 ``dynamic_axes`` 混用），避免约束冲突。
        """
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        num_layers = int(self.expert_config.num_hidden_layers)
        prefix_len = 968

        logger.info("Start export denoise onnx ...")
        print(colored("Start Pi05 denoise (Pi05DenoiseStep) export onnx...", "green"))

        prefix_pad_masks = torch.ones(
            (1, prefix_len), dtype=torch.bool, device="cuda"
        )
        past_keys = []
        past_values = []
        for _ in range(num_layers):
            past_keys.append(
                torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            )
            past_values.append(
                torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            )
        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)

        x_t = torch.randn(
            (1, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device="cuda",
        )
        timestep = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        output_path = f"{output_dir}/denoise.onnx"
        export_kw: dict = {}
        if dynamo:
            from torch.export import Dim

            batch_dim = Dim("batch", min=1, max=4096)
            prefix_seq_dim = Dim("prefix_seq", min=1, max=4096)
            export_kw["dynamic_shapes"] = {
                "prefix_pad_masks": {0: batch_dim, 1: prefix_seq_dim},
                "past_keys": {1: batch_dim, 2: prefix_seq_dim},
                "past_values": {1: batch_dim, 2: prefix_seq_dim},
                "x_t": {0: batch_dim},
                "timestep": {0: batch_dim},
            }
        else:
            export_kw["dynamic_axes"] = {
                "prefix_pad_masks": {0: "batch_size", 1: "prefix_seq_len"},
                "past_keys": {1: "batch_size", 2: "prefix_seq_len"},
                "past_values": {1: "batch_size", 2: "prefix_seq_len"},
                "x_t": {0: "batch_size"},
                "timestep": {0: "batch_size"},
                "v_t": {0: "batch_size"},
            }

        with torch.inference_mode():
            torch.onnx.export(
                self,
                (prefix_pad_masks, past_keys_tensor, past_values_tensor, x_t, timestep),
                output_path,
                export_params=True,
                input_names=[
                    "prefix_pad_masks",
                    "past_keys",
                    "past_values",
                    "x_t",
                    "timestep",
                ],
                output_names=["v_t"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                **export_kw,
            )
        _patch_denoise_onnx_float_bias_to_bfloat16(output_path)
        _patch_denoise_onnx_prefix_proj_float_bias_for_bf16_matmul_add(output_path)
        _patch_denoise_onnx_gemma_bf16_bias_to_float_if_fp8(output_path)
        end = time.time()
        logger.info("export onnx to %s done cost:%ss", output_dir, end - start)
        print(
            colored(
                f"Pi05 denoise export onnx done to {output_path} cost:{end - start}s",
                "green",
            )
        )
        return self

    def quantize(self, quant_cfg, calib_data, export_dir, *, measure_quant_error: bool = False):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        # FP8_KV_CFG / *_bmm_quantizer：ModelOpt 要求 ``mtq.quantize`` 的根模块为 HF PreTrainedModel，
        # 与 LLM.quantize(self.model, ...) 一致；标定仍走完整 ``forward`` 以覆盖 action/time 投影与 expert。
        if quant_config_targets_hf_bmm_kv(quant_cfg):
            quantize_model(
                self.gemma_expert,
                quant_cfg,
                calib_dataloader,
                forward_context=self,
                measure_quant_error=measure_quant_error,
            )
        else:
            quantize_model(
                self,
                quant_cfg,
                calib_dataloader,
                measure_quant_error=measure_quant_error,
            )
        self.is_quantized = True
        set_dynamic_quant(self, "bf16")

        self.export(export_dir, dynamo=False)
        onnx_path = f"{export_dir}/denoise.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)
        _patch_denoise_onnx_float_bias_to_bfloat16(onnx_path)
        _patch_denoise_onnx_prefix_proj_float_bias_for_bf16_matmul_add(onnx_path)
        _patch_denoise_onnx_gemma_bf16_bias_to_float_if_fp8(onnx_path)

    def _wrap_past_key_values(
        self, past_keys: torch.Tensor, past_values: torch.Tensor
    ) -> DynamicCache:
        k_v_cache = DynamicCache()
        num_layers = past_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(past_keys[i : i + 1], past_values[i : i + 1], i)
        return k_v_cache

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, _ATTN_MASK_FILL_VALUE)

    def _embed_suffix_pi05(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """pi05：时间 adaRMS 条件 + action_in_proj，与 openpi embed_suffix（pi05 分支）一致。"""
        if timestep.ndim != 1:
            raise ValueError(f"timestep must be 1D (batch,), got shape {tuple(timestep.shape)}")
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        # 与 ``action_in_proj`` / ``time_mlp_*`` 的权重 dtype 对齐；标定数据里 ``x_t``/``timestep`` 可能为
        # bf16（``_calib_batch_to_device_only`` 保留 dtype），而投影层常为 fp32，否则 ``F.linear`` 报
        # ``mat1 and mat2 must have the same dtype``.
        mlp_w_dtype = self.time_mlp_in.weight.dtype
        time_emb = time_emb.to(dtype=mlp_w_dtype)

        action_emb = self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        action_time_emb = action_emb
        action_time_dim = action_time_emb.shape[1]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks = action_time_mask

        att_base = self._suffix_ar_mask.to(device=device).expand(bsize, -1)
        return action_time_emb, pad_masks, att_base, adarms_cond

    def forward(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prefix_pad_masks: bool/float [batch, prefix_len]，与 LLM prefix 一致。
            past_keys: [num_layers, batch, num_kv_heads, prefix_len, head_dim]（与 LLM 导出堆叠方式一致）。
            past_values: 与 past_keys 相同布局。
            x_t: float [batch, action_horizon, action_dim]。
            timestep: float [batch]，与 openpi sample 中 expanded_time 一致。

        Returns:
            v_t: float32 [batch, action_horizon, action_dim]。
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix_pi05(
            x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, dtype=torch.int64)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1, dtype=torch.int64) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        expert_dtype = self.gemma_expert.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)

        past = self._wrap_past_key_values(past_keys, past_values)
        outputs = self.gemma_expert(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=suffix_embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = outputs.last_hidden_state
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @classmethod
    def construct_model(cls, pi05_model, dtype: torch.dtype | None = None):
        """
        从 Pi05Model 包装器或已加载的 PI0Pytorch（policy._model）构建，共享子模块权重引用。
        """
        if not getattr(pi05_model.config, "pi05", False):
            raise ValueError("Pi05DenoiseStep 仅支持 config.pi05 is True 的 PI0Pytorch 模型")

        gemma_expert = pi05_model.paligemma_with_expert.gemma_expert.model
        expert_config = pi05_model.paligemma_with_expert.gemma_expert.config

        if dtype is not None:
            logger.debug("construct_model dtype=%s ignored (weights keep loaded dtype)", dtype)

        return cls(
            gemma_expert=gemma_expert,
            expert_config=expert_config,
            action_in_proj=pi05_model.action_in_proj,
            time_mlp_in=pi05_model.time_mlp_in,
            time_mlp_out=pi05_model.time_mlp_out,
            action_out_proj=pi05_model.action_out_proj,
            action_horizon=pi05_model.config.action_horizon,
            action_dim=pi05_model.config.action_dim,
        )

    @classmethod
    def construct_from_name_path(cls, model_name: str, model_path: str):
        from .model_pi05 import Pi05Model

        wrapper = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(wrapper, dtype=torch.bfloat16)
