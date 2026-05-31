# Pi0.5 LLM（PaliGemma 解码器，prefill 阶段）**延时优先** NVFP4 量化配置。
#
# 目标平台：Jetson Thor / Blackwell（具备 FP4 张量核，吞吐 ≈ 2×FP8 ≈ 4×BF16）。
# LLM 是 prefix/prefill（S≈968，compute-bound），延时由大 GEMM 主导，故策略是
# **最大化 FP4 GEMM 覆盖、不留精度孤岛**：
#
#   - 全部 Linear（q/k/v/o + gate/up/down，或 fused 后的 gate_up_proj）走 **NVFP4 W4A4**；
#     与 ``llm_quant_nvfp4_fp8_mix_cfg.py`` 的关键区别：**不**把任何 MLP 层降级为 FP8
#     （降级会变慢且打断统一 FP4 kernel 选择）。
#   - 注意力：FP4 不跑 attention kernel，``*q/k/v_bmm`` 用 **FP8**（Thor FmhaD256 插件路径）。
#   - KV cache：**FP8**（压 KV 带宽，利于下游 expert 交叉注意力与显存）。
#   - layernorm：Gemma RMSNorm 无 Linear 子模块，无量化器（见 add_nvfp4_input_layernorm_explicit 注释）。
#
# 建议搭配：feature_config 开 ``fused_mlp``（gate/up 合并成单个 NVFP4 大 GEMM），并走 cutedsl
# FmhaD256 插件注意力路径，延时最佳。
#
# 精度兜底：若纯 W4A4 精度不达标，**优先换 awq_lite 算法**（见文件末尾开关）而不是把层降 FP8——
# awq 的 per-channel pre_quant_scale 折进权重、运行期≈0 开销，可在保住满 FP4 覆盖的同时恢复精度。
import copy

from modelopt.torch.quantization import FP8_KV_CFG, NVFP4_DEFAULT_CFG

from model_optimizer.quantization.cfg import add_nvfp4_input_layernorm_explicit

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
_qc = QUANT_CFG["quant_cfg"]

# 全局即 NVFP4 W4A4（``*weight_quantizer`` / ``*input_quantizer``），不做任何 per-layer FP8 降级。
add_nvfp4_input_layernorm_explicit(_qc)

# 注意力 BMM 走 FP8（E4M3，per-tensor）——FP4 不适用于 attention kernel。
_FP8_BMM = {"num_bits": (4, 3), "axis": None, "enable": True}
_qc["*q_bmm_quantizer"] = dict(_FP8_BMM)
_qc["*k_bmm_quantizer"] = dict(_FP8_BMM)
_qc["*v_bmm_quantizer"] = dict(_FP8_BMM)

# KV cache FP8（压带宽）。FP8_KV_CFG 里是 *output_quantizer 等键，update 合并即可。
_qc.update(FP8_KV_CFG["quant_cfg"])

# 校准算法：max 最快产出且对 NVFP4 通常足够。
QUANT_CFG["algorithm"] = "max"

# ── 精度兜底开关（默认关闭，按需打开其一）────────────────────────────────────────
# 若纯 W4A4(max) 精度不够，改用 awq_lite：不降级任何层、保住满 FP4 覆盖，运行期≈0 开销。
# QUANT_CFG["algorithm"] = {"method": "awq_lite", "alpha_step": 0.5}
#
# 或只对“最敏感的少数层”兜底（最小化延时代价；通常是首/末层或 down_proj）。示例：
# _FP8_LINEAR = {"num_bits": (4, 3), "axis": None}
# for i in (0, 17):
#     for sub in ("mlp.gate_up_proj", "mlp.down_proj"):
#         _qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_LINEAR)
#         _qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)
