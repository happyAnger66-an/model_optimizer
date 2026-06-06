# C4→C5：`gate_res_adarms_fp8_static_fp16` 算子说明

> **实现**：`third_party/FlashRT/csrc/kernels/decoder_fused.cu::gate_res_adarms_fp8_static_fp16_kernel`  
> **Host 入口**：`gate_res_adarms_fp8_static_fp16`（`<<<S, 256, shmem>>>`）  
> **Python 绑定**：`fvk.gate_res_adarms_fp8_static_fp16`（`flashrt_decoder/kernels.py` 加载）  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` — C4→C5（每层 attention 块末尾）  
> **复用**：同 kernel 亦用于 C7→C1_next（层间过渡，style 换为下一层 `sa_ptr`）  
> **相关文档**：[`fused_adarms_fp8_static_fp16.md`](fused_adarms_fp8_static_fp16.md)（C1，attn 前 AdaRMS）  
> **下游 GEMM**：[`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md) §5 槽位 k=2（gate+up）

---

## 1. 功能概述

FlashRT **C4→C5 融合 kernel** 将 OpenPI/Pi0.5 decoder 中 self-attention **结束之后、FFN 开始之前** 的三步合并为一次 GPU 调用：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | **Gated residual** | `x ← x + fg ⊙ gate`（第一次残差连接） |
| 2 | **AdaRMSNorm** | `y = RMSNorm(x) * (1 + scale) + shift`（`post_attention_layernorm`） |
| 3 | **静态 FP8 量化 + gate 更新** | `xn_fp8 = quantize(y)`；写出新的 `gate` 供 C7 第二次残差使用 |

PyTorch/OpenPI 中对应逻辑（Expert 分支，`gemma_pytorch.py`）：

```python
out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])   # C4 → fg

# first residual
out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
# x + o_proj_out * gate，gate 来自本层 input_layernorm

after_first_residual = out_emb.clone()
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
# 随后 layer.mlp(out_emb) → C5/C6
```

`_gated_residual` 语义（`modeling_gemma.py`）：

```python
return x + y * gate   # gate 为 None 时退化为 x + y
```

FlashRT 一次 kernel 完成 gated residual + post-attention norm + FP8 量化，norm 结果以 **FP8** 形式直接进入 C5 gate+up GEMM，避免 `[S, D]` fp16 中间激活写回 DRAM。

---

## 2. 在 Denoise 流水线中的位置

每个扩散步 `s` 执行 18 层 Gemma Expert，C4→C5 位于每层 attention 子块与 FFN 子块之间：

```text
扩散步 s:
  for layer l in 0..17:
    C1   fused_adarms_fp8_static_fp16        → xn_fp8, gate（input gate）
    C2   fp8_gemm_descale_fp16  (k=0)      → qkv
    C2b  qkv_split_rope_kvcache_fp16
    C3   Cross-attention                     → attn_out
    C4   quantize_fp8 + fp8_gemm_descale_fp16 (k=1) → fg   # o_proj
    ★ C4→5 gate_res_adarms_fp8_static_fp16  → xn_fp8, gate（new gate）
    C5   fp8_gemm_descale_fp16  (k=2)      → fg           # gate+up
    C6   gate_geglu_merged + fp8_gemm_descale_fp16 (k=3) → fg  # down
    C7   gate_res_adarms（下一层 C1）或 gate_res_fp16（最后一层）
```

对应 `denoise_step` → `paligemma_with_expert.forward(..., suffix_embs)` 中 **Expert 第 `l` 层的 attention 块结束到 `layer.mlp()` 开始** 之间的逻辑。

### 2.1 pipeline 调用示例

```python
act_scale_gu = act_scales + (l * 4 + 2) * 4
fvk.gate_res_adarms_fp8_static_fp16(
    fg, gate, x, sf_ptr,      # gemm_out, prev_gate, residual, style
    xn_fp8, gate,              # fp8_out, gate_out（gate 原地覆盖）
    S, D, act_scale_gu, stream,
)
```

---

## 3. 张量布局

| 参数 | pipeline 名 | 形状 | dtype | 含义 |
|------|-------------|------|-------|------|
| `gemm_out` | `fg` | `[S, D]` | fp16 | C4 O-proj GEMM 输出 |
| `prev_gate` | `gate`（输入） | `[S, D]` | fp16 | C1 写出的 gate（`input_layernorm` 第三段） |
| `residual` | `x` | `[S, D]` | fp16 | 残差流 hidden states，**原地更新** |
| `style` | `sf_ptr` | `[S, 3*D]` | fp16 | 预计算 Ada 调制：`post_attention_layernorm.dense(cond)` |
| `fp8_out` | `xn_fp8` | `[S, D]` | fp8 (E4M3) | 量化后的 norm 输出，供 C5 gate+up FP8 GEMM |
| `gate_out` | `gate`（输出） | `[S, D]` | fp16 | 新 gate（`post_attention_layernorm` 第三段），供 C7 第二次残差 |
| `descale_ptr` | `act_scale_gu` | 标量（device） | fp32 | gate+up 路径静态 `act_scale`，槽位 `l*4+2` |

- `S`：action token 数（Pi0.5 通常为 10）
- `D`：隐藏维（≈1024）
- `style` 每行布局：`sc = style[r*3*D : r*3*D+D]`，`sh = sc+D`，`gt = sh+D`

`sf_ptr` 由 `precompute.py::precompute_adarms_styles` 离线预计算：

```
sf[s,l,:] = time_emb @ post_attention_layernorm.dense.weight.T + bias   # 每 token 一行，长度 3*D
```

---

## 4. 数学公式

对每一 token 行 `r`、隐藏维 `i`：

### 4.1 Step 1 — Gated residual（第一次残差）

```
x[i] ← x[i] + fg[i] * gate_input[i]
```

- `gate_input` 是 C1 `fused_adarms_fp8_static_fp16` 在 **attn 前** 从 `sa_ptr` 取出的 gate，经 `bufs['gate']` 保留至 C4 之后
- 更新后的 `x` 写回 `residual` buffer（原地），并作为后续 RMSNorm 的输入

OpenPI 等价：

```python
out_emb = _gated_residual(hidden_states, o_proj_out, gate_from_input_ln)
```

### 4.2 Step 2 — AdaRMSNorm（post-attention）

```
rms  = sqrt(mean(x²) + ε),   ε = 1e-6
y[i] = x[i] / rms * (1 + scale[i]) + shift[i]
```

- `scale / shift` 来自 `sf_ptr` 的前两段（`post_attention_layernorm`）
- 实现中用 `rstd = rsqrt(mean(x²) + ε)`，`y = x * rstd * (1+scale) + shift`

OpenPI 等价（`GemmaRMSNorm.forward`，`cond=adarms_cond`）：

```python
modulation = self.dense(cond)           # Linear(Da → 3*D)
scale, shift, gate_new = chunk(modulation, 3)
normed = rmsnorm(x) * (1 + scale) + shift
```

### 4.3 Step 3 — 静态 FP8 量化 + gate 更新

```
inv_scale = 1 / max(act_scale_gu, 1e-12)
xn_fp8[i] = clamp(y[i] * inv_scale, -448, 448)   →  __nv_fp8_e4m3
gate[i]   = gate_new[i]                             # 覆盖 C1 留下的旧 gate
```

- **E4M3** 最大有限值 ≈ **448**；`act_scale_gu` 须与 C5 GEMM 的 `act_descale` **相同**（槽位 `l*4+2`）
- 新 `gate` 持久化到 `bufs['gate']`，供 C7 `gate_res_adarms` / `gate_res_fp16` 做 FFN 后的第二次 gated residual

---

## 5. 计算三阶段

Kernel 对每个 row（`blockIdx.x = r`）分三阶段执行（与 C1 结构类似，Phase 1 多了 gated residual）：

```text
Phase 1: x ← x + fg*gate；各线程 stride 累加 sum(x²)
    ↓
Phase 2: warp/block 两级归约 → sum_sq → rstd
    ↓
Phase 3: 归一化 + Ada 仿射 + FP8 量化 + 写新 gate
```

| 阶段 | 操作 | 关键技术 |
|------|------|----------|
| **Phase 1** | `res = x + fg*prev_gate`；写回 `x`；累加 `sum(res²)` | 融合 residual 与 norm 的 `sum_sq` 预计算，单次遍历 `D` |
| **Phase 2** | 全 block 归约得到 `sum_sq` | `__shfl_xor_sync` 蝶形归约 + shared memory 跨 warp 归约（同 C1） |
| **Phase 3** | `normed = x*rstd*(1+scale)+shift` → FP8；写 `gate_new` | block 级预计算 `inv_scale`；E4M3 clamp |

核心 loop（Phase 1 + 3 摘录）：

```cpp
// Phase 1
float res = residual[r*D+i] + gemm_out[r*D+i] * prev_gate[r*D+i];
residual[r*D+i] = res;
sum_sq += res * res;

// Phase 3
float v = residual[r*D+i] * rstd;
float normed = v * (1 + scale[i]) + shift[i];
fp8_out[r*D+i] = clamp(normed * inv_scale, E4M3);
gate_out[r*D+i] = gate_new[i];
```

---

## 6. OpenPI ↔ FlashRT 对照

| OpenPI（pi05 denoise，Expert 层 `l`） | FlashRT |
|---------------------------------------|---------|
| `o_proj(att_output)` | C4 `quantize_fp8` + `fp8_gemm_descale_fp16` → `fg` |
| `_gated_residual(x, o_proj_out, gate_C1)` | Phase 1：`x += fg * gate` |
| `post_attention_layernorm(x, cond=adarms_cond)` | Phase 2–3：RMSNorm + `sf_ptr` scale/shift |
| （无，FP8 推理特有） | Phase 3：量化到 `xn_fp8`，`act_scales[l*4+2]` |
| 返回新 `gate` | 写回 `gate`，C7 第二次残差使用 |
| `layer.mlp(out_emb)` | C5 gate+up GEMM + C6 GeGLU + down GEMM |

Gate 的生命周期（单层内）：

```text
C1 input_layernorm  → gate 写出（attn 前调制 gate）
         ↓ 保留
C4→C5 第一次残差    → 消费 gate_input：x += fg * gate_input
         ↓ 覆盖
C4→C5 post_attn     → gate 更新为 gate_new
         ↓ 保留
C7 第二次残差       → 消费 gate_new：x += down_out * gate_new
```

---

## 7. 与 C1 的对比

两者均为 **AdaRMSNorm + FP8 quant + gate 写出** 的融合模式，但职责与上下游不同：

| | **C1 `fused_adarms_fp8_static_fp16`** | **C4→C5 `gate_res_adarms_fp8_static_fp16`** |
|---|---|---|
| OpenPI 对应 | `input_layernorm` | `_gated_residual` + `post_attention_layernorm` |
| style 来源 | `sa_ptr`（attn 前） | `sf_ptr`（attn 后 / FFN 前） |
| 输入 hidden | 层入口 `x`（无前置 GEMM residual） | 先 `x += fg * gate`，再 norm |
| 上游 | 无 GEMM | C4 O-proj 输出 `fg` |
| 下游 | C2 QKV GEMM | C5 gate+up GEMM |
| FP8 scale 槽位 | `l*4+0` | `l*4+2` |
| gate 用途 | 供 C4→C5 第一次残差 | 供 C7 第二次残差 |

---

## 8. C7 复用说明

同一 kernel 在 **C7→C1_next** 再次调用（非最后一层）：

```python
fvk.gate_res_adarms_fp8_static_fp16(
    fg, gate, x, sa_next_ptr,   # gemm_out=down GEMM 输出；style=下一层 input_layernorm
    xn_fp8, gate, S, D, act_scale_next, stream,
)
```

语义：FFN down 输出后的 **第二次 gated residual** + **下一层 input_layernorm** + FP8 量化。  
最后一层（`l == layers-1`）仅做 `gate_res_fp16`（无 norm/量化），Final AdaRMS 由 `adarms_fp16` 单独处理。

---

## 9. 融合动机

| 若拆开（PyTorch 语义） | 问题 |
|------------------------|------|
| `gate_res_fp16` + `adarms_fp16` + `quantize_fp8_static_fp16` | 3 次 kernel launch；`x` 与 norm 中间结果多次 DRAM 读写 |
| 融合为一步 | 1 次 launch；Phase 1 内联 residual 与 `sum_sq`；norm 结果直接写 FP8 |

与 C1 相同，本 kernel 是 **memory-bound 元素级链 + 一次 reduction** 的典型融合候选；后面的 C5 GEMM 为 compute-bound，融合收益主要在 attention/FFN 边界省带宽与 launch。
