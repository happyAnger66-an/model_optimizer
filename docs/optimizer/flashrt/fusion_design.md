# FlashRT Decoder 融合算子设计（`fusion_design.md`）

> 实现：`FlashRT` 构建产物 `flash_rt_kernels*.so`（仓内经 `flashrt_decoder/kernels.py` 加载为 `fvk`）  
> 编排：`src/model_optimizer/infer/native/flashrt_decoder/pipeline.py::decoder_forward`  
> 缓冲：`flashrt_decoder/driver.py::DecoderBuffers`（含 `gate`、`xn_fp8` 等）  
> OpenPI 对照：`third_party/openpi/.../pi0_pytorch.py`、`gemma_pytorch.py`、`modeling_gemma.py`

与 [`dec_rope.md`](dec_rope.md)（RoPE 预计算）、[`../ddup/flashrt_denoise_fp8.md`](../ddup/flashrt_denoise_fp8.md)（FP8 标定）并列：
本文说明 **为何把多步 PyTorch op 合成一个 kernel**、**如何选取融合边界**，并以 C1 `fused_adarms_fp8_static_fp16` 与 **`gate` buffer** 为例。

---

## 1. 功能概述

Pi0.5 denoise 的 FlashRT 路径把 Expert Gemma **18 层 × 10 扩散步** 展开成固定形状的 CUDA kernel 序列。
其中大量算子在 PyTorch 里是独立的（RMSNorm、Ada 调制、量化、RoPE、残差），在 FlashRT 里按 **带宽型小块** 与 **算力型大块（GEMM/FMHA）** 交替融合。

典型动机：

| 动机 | 说明 |
|------|------|
| **减 DRAM 往返** | 避免 `[S,D]` fp16 中间激活写回再读（`S=10`，`D`≈1024，但 launch 次数极多） |
| **减 kernel launch** | 每层每步 10+ 次小 op，融合可砍掉多次 launch 开销 |
| **对齐 FP8 入口** | 下一级 Tensor Core GEMM 需要 FP8 输入 → 在 **GEMM 前** 把 Norm+Quant 融在一起 |

**不是**「能融尽融」：大 GEMM、Attention 仍独立；融合边界与 **静态 `act_scales` 槽位** 一一对应（每层 4 点，见 FP8 文档 §2.1）。

---

## 2. 整条 decoder 的数据流（融合块 vs 独立块）

```text
每扩散步 s（≡ openpi denoise_step 一次）:
  action_in_proj     → gmm_fp16 + add_bias_fp16          [独立，≡ embed_suffix]
  for layer l:
    C1  fused_adarms_fp8_static_fp16                   [融合: AdaRMS + FP8 quant + 写 gate]
    C2  fp8_gemm_descale (QKV)                         [算力核]
    C2b qkv_split_rope_kvcache_fp16                     [融合: split + RoPE + 写 KV]
    C3  attention / FMHA                                [算力核]
    C4  quantize_fp8 + fp8_gemm_descale (O)            [量化 + GEMM]
    C4→5 gate_res_adarms_fp8_static_fp16               [融合: gated residual + post AdaRMS + quant]
    C5  fp8_gemm_descale (merged gate+up)              [算力核]
    C6  gate_geglu_merged_fp8_fp16 + down GEMM         [融合 GeGLU + 算力核]
    C7→1 gate_res_adarms 或 gate_res_fp16              [层间 residual + 下一层/末层 Ada]
  Final adarms_fp16 + action_out_proj (gmm + bias)     [≡ action_out_proj + Euler 写回 noise]
```

启发式（选取「融哪些」）：

1. **GEMM 入口前**：融 `Norm + 静态 FP8 量化`（C1、C4→C5、层间 C7→C1）。
2. **GEMM epilogue**：融 `descale`（`fp8_gemm_descale`）、GeGLU 的 `SiLU(gate)×up`（C6）。
3. **KV 路径**：融 `QKV layout + RoPE + 写 Kc/Vc`（C2b，见 [`dec_rope.md`](dec_rope.md)）。
4. **跨步不变子图**：提出预计算（`sa/sf/fs` Ada 调制、`rope`），运行时 kernel 只 **应用**。

---

## 3. 融合设计的一般原则

### 3.1 从数据流切口切，不从函数名切

先画固定推理图（pi05：`S`、`D`、`layers`、`steps` 恒定），标出：

```text
[带宽型: norm / scale / quant / rope / gated-add / SiLU×mul]
        ↓ 尽量少落盘
[算力型: GEMM / FMHA]
        ↓ epilogue 可再融 descale、bias
[下一带宽型 …]
```

**适合融合**：逐元素、算术强度低、输出立刻被下一 op 消费、形状可特化。  
**通常不融主体**：大 GEMM、FMHA；仅 CUTLASS 式 **epilogue 融合**（如 `fp8_gemm_descale`）。

### 3.2 预计算 vs 运行时融合

| 内容 | 策略 | 例子 |
|------|------|------|
| 随扩散步变、层间共享少 | 预计算 buffer | `time_mlp` → `sa/sf/fs`（`precompute_adarms_styles`） |
| 随 token 位置变 | 预计算表 | `build_dec_rope` → `weights['rope']` |
| 每 token 每步即时算 | 融进 kernel | RMSNorm + 读 `sa_ptr` 的 scale/shift/gate |

不把 `dense(adarms_cond)` 放进 C1 runtime：已在 setup 时展开为 `sa`，减少 kernel 内 MLP 依赖。

### 3.3 权衡

| 收益 | 代价 |
|------|------|
| 更低带宽、更少 launch | 调试难；与 OpenPI 逐 op 对数需更细 |
| 固定形状极致调优 | 改 `S`/`D`/层数需重编译或新特化 |
| FP8 端到端 | 每个融合边界单独标定 `act_scales`；边界选错影响精度与性能 |
| `sa` 预计算 | 占显存；换 prompt/步数需重建 |

---

## 4. 案例：C1 `fused_adarms_fp8_static_fp16`

### 4.1 融了什么

OpenPI（pi05）单层 Expert 在 self-attn **之前**：

```python
# gemma_pytorch.py — 仅 expert 分支 i=1
hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

# modeling_gemma.py — GemmaRMSNorm
modulation = self.dense(cond)           # Linear(Da → 3*D)
scale, shift, gate = chunk(modulation, 3)
x_norm = rmsnorm(x) * (1 + scale) + shift
# 随后 q_proj / k_proj / v_proj(x_norm)
```

FlashRT C1 **一次 kernel** 完成：

1. 读 `x`（残差流，`[S,D]` fp16）与 `sa_ptr`（预计算的 `3×D` 调制，含 scale/shift/gate）
2. RMSNorm + Ada 仿射（用 scale/shift）
3. 按 `act_scale_qkv` **静态量化** → `xn_fp8`（供 C2 QKV GEMM）
4. **写出 `gate`** → `bufs['gate']`（`[S,D]` fp16）

`act_scale_qkv = act_scales + (l*4+0)*4` 仅 FP8 路径存在，OpenPI 无对应项。

### 4.2 「写 gate」指什么

**不是** FFN 里 `gate_proj` 的 GEGLU gate（那是 C6 `gate_geglu_merged`）。

**是** AdaRMS 调制向量里的 **第三段 `gate`**：`dense(cond)` 切成 `(scale, shift, gate)` 后的 `gate`，形状与隐藏维同维 **`[S, D]`**（每 action token 一行；`sa` 预计算时每行后 `D` 维即 gate）。

| 分量 | C1 内 | 后续用途 |
|------|-------|----------|
| scale / shift | 当场用于归一化后仿射 | 结果进入 `xn_fp8`，**不**以完整 fp16 `[S,D]` 落盘 |
| **gate** | **写入 `bufs['gate']`** | C4→C5 **第一次 gated residual** |

为何要落盘：C1 到 C4 之间还有 QKV、RoPE、KV cache、Attention、O-proj，**隔多个 kernel** 才用到 gate，无法只留在寄存器。

OpenPI 用法（第一次残差，attn 之后）：

```python
# _gated_residual(x, y, gate)  →  x + y * gate
out_emb = _gated_residual(hidden_states, out_emb, gates[i])
```

FlashRT 对应 **C4→C5** `gate_res_adarms_fp8_static_fp16(fg, gate, x, sf_ptr, …)`：

- `x`：进本层前的残差（C1 前写入的 `x`，attn 期间未改）
- `fg`：O-proj 输出
- `gate`：C1 写下的张量 → 语义 **`x ← x + fg * gate`**

时间线：

```text
C1: input_layernorm → xn_fp8；gate 写入 buffer
      ↓
C2–C4: QKV → Attn → O-proj → fg
      ↓
C4→5: x + fg * gate（再叠 post-attn AdaRMS 等，同一融合核内完成）
```

同一 `gate` buffer 在层内会被 **覆盖**：`post_attention_layernorm` 产生的新 gate 在 C4→C5 / C7→下一层 C1 写入，供 **第二次** gated residual（FFN 后）。C1 写的 gate **只服务本层 attn 后那一次**。

### 4.3 与「不写 fp16 中间激活」的关系

- **Norm 结果**：紧接 QKV → 融进 C1 并输出 **`xn_fp8`**，避免 `[S,D]` fp16 写 DRAM。
- **gate**：必须写 **`gate` buffer**，属于 **有意落盘**，与「省掉 norm 输出落盘」是两件不同的事。

---

## 5. OpenPI ↔ FlashRT 对照表

| FlashRT | OpenPI（pi05） |
|---------|----------------|
| `gmm_fp16` + `add_bias_fp16`（步初） | `embed_suffix` → `action_in_proj(x_t)` |
| `time_mlp` → 预计算 `sa/sf/fs` | `embed_suffix` → `adarms_cond`；各层 `GemmaRMSNorm(..., cond=...)` |
| C1 `fused_adarms_*` | `layer.input_layernorm` + 为 QKV 准备归一化激活；**gate 暂存** |
| C2 `fp8_gemm_descale` | `q_proj/k_proj/v_proj` |
| C2b `qkv_split_rope_kvcache` | RoPE + 写 KV（[`dec_rope.md`](dec_rope.md)） |
| C3 attention | `eager_attention_forward` / FMHA |
| C4 O GEMM | `o_proj` |
| C4→5 `gate_res_adarms_*` | `_gated_residual` + `post_attention_layernorm` |
| C5–C6 FFN FP8 路径 | `mlp`（gate/up/down + GeGLU） |
| C7 / Final `adarms` / `gate_res` | 层间与 final norm + 第二次 residual |
| 步末 `gmm` + `add_bias` → `noise` | `action_out_proj` + Euler 更新 `x_t` |

代码内块注释：`pipeline.py` 步初 `action_in_proj`、每层 C1 等处已与上述路径交叉引用。

---

## 6. 每层四个 FP8 量化点与融合边界

与 [`../ddup/flashrt_denoise_fp8.md`](../ddup/flashrt_denoise_fp8.md) §2.1 一致：

| 槽位 `l*4+k` | 名称 | 融合边界（激活进入 FP8 GEMM 前） |
|--------------|------|----------------------------------|
| k=0 | qkv | C1 `fused_adarms_*` |
| k=1 | o_proj | C4 `quantize_fp8`（与 O GEMM 分离，因 attn 输出布局 `[S*NH, HD]`） |
| k=2 | gate_up | C4→5 `gate_res_adarms_*` |
| k=3 | down | C6 `gate_geglu_merged_*` |

标定时必须按 **融合后的张量语义** 测 amax，而不是按 PyTorch 未融合的中间名随意对齐。

---

## 7. 相关文件

| 文件 | 角色 |
|------|------|
| `pipeline.py` | `decoder_forward` 融合序列与 OpenPI 对照注释 |
| `driver.py` | `DecoderBuffers`（`gate`、`xn_fp8`、`x`、`fg` 等） |
| `precompute.py` | `precompute_adarms_styles`（`sa/sf/fs`） |
| `kernels.py` | 加载 `fvk`（`fused_adarms_*`、`gate_res_*` 等） |
| `dec_rope.md` | suffix RoPE 预计算与 C2b |
| `flashrt_denoise_fp8.md` | FP8 标定与四层四量化点 |
| `fp8_gemm_descale_fp16.md` | C2/C4/C5/C6 GEMM + descale 语义与 OpenPI 对照 |
| `native_decoder_implementation_todo.md` §8 | 端到端契约与 Thor 验证 |

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-02 | 初版：融合选取原则、decoder 数据流、C1/gate 详解、OpenPI 对照 |
