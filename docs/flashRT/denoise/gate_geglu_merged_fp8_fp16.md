# C6：`gate_geglu_merged_fp8_fp16` 算子说明

> **Python 绑定**：`fvk.gate_geglu_merged_fp8_fp16`  
> **生产实现**：`third_party/FlashRT/csrc/kernels/activation.cu::gate_silu_mul_merged_fp8_fp16`  
> **等价参考**：`third_party/FlashRT/csrc/kernels/decoder_fused.cu::geglu_fp8_static_fp16`（Pi0.5 同等数学，decoder 专用副本）  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` — C6 前半（每层 FFN 激活段）  
> **量化槽位**：`k=3`（`act_scales[l*4+3]`，与下游 down GEMM descale **共用**）  
> **上游**：[`fp8_gemm_descale_fp16_c5.md`](fp8_gemm_descale_fp16_c5.md)（C5 gate+up GEMM → `fg`）  
> **下游**：C6 down GEMM（[`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md) §5 槽位 k=3）

---

## 1. 功能概述

FlashRT **C6 GeGLU 融合 kernel** 将 OpenPI/Pi0.5 decoder FFN 中 **gate/up Linear 之后、down Linear 之前** 的三步合并为一次 GPU 调用：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | **GELU（tanh 近似）** | 作用于 gate 分支线性输出 |
| 2 | **逐元素乘** | `GELU(gate) × up`（GeGLU 模式） |
| 3 | **静态 FP8 量化** | 写入 `hid_fp8`，供 down GEMM 使用 |

pipeline 调用（注释写 SiLU，Pi0.5 实际为 **GELU tanh 近似**）：

```python
act_scale_down = act_scales + (l * 4 + 3) * 4
fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, act_scale_down, stream)
```

OpenPI 等价（`GemmaMLP.forward`，`modeling_gemma.py`）：

```python
def forward(self, x):
    down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           本 kernel 对应 act_fn(gate_lin) * up_lin（不含 down_proj）
```

Pi0.5 action expert 的 `hidden_activation` 为 **`gelu_pytorch_tanh`**（非 SiLU）。Python 名含 `geglu`、历史实现名含 `silu_mul`，运行时均走 **GELU tanh 近似** 分支。

---

## 2. 在 Denoise 流水线中的位置

```text
扩散步 s:
  for layer l in 0..17:
    ...
    C5   fp8_gemm_descale_fp16 (k=2)      → fg [S, 2H] fp16   # gate_proj + up_proj
    ★ C6 gate_geglu_merged_fp8_fp16       → hid_fp8 [S, H]    # GELU(gate) × up
    C6   fp8_gemm_descale_fp16 (k=3)      → fg fp16           # down_proj
    C7   gated residual + 下一层 norm
```

对应 `denoise_step` → `paligemma_with_expert.forward(..., suffix_embs)` 中 **Expert 第 `l` 层 `layer.mlp()` 的中间段**。

`gemma_pytorch.py` 侧：

```python
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
out_emb = layer.mlp(out_emb)
```

| OpenPI | FlashRT |
|--------|---------|
| `gate_proj(x)` + `up_proj(x)` | C5 合并 GEMM → `fg` |
| `act_fn(gate_lin) * up_lin` | **本 kernel → `hid_fp8`** |
| `down_proj(hidden)` | C6 down GEMM（下一行） |

---

## 3. 张量布局

| 参数 | pipeline 名 | 形状 | dtype | 含义 |
|------|-------------|------|-------|------|
| `merged` | `fg` | `[S, 2H]` | fp16 | C5 输出：`[gate_linear \| up_linear]` |
| `out` | `hid_fp8` | `[S, H]` | fp8 (E4M3) | GeGLU 结果，down GEMM 激活输入 |
| `seq` | `S` | int | — | action token 数（Pi0.5 通常 10） |
| `half_dim` | `H` | int | — | FFN 中间维（`intermediate_size`） |
| `d_scale` | `act_scale_down` | device 标量 | fp32 | `act_scales[l*4+3]` |

`fg` 行内布局（token `s`）：

```text
fg[s, 0:H]   → gate 线性输出 gv（GELU 前）
fg[s, H:2H]  → up  线性输出 uv
```

索引：`merged[s * 2*H + h]` = gate，`merged[s * 2*H + H + h]` = up。

---

## 4. 数学公式

对每一 `(s, h)`：

### 4.1 GELU（tanh 近似，Pi0.5 公式）

OpenPI `gelu_pytorch_tanh` 等价形式：

```
gelu(gv) = gv / (1 + exp(-1.5957691216057308 * gv * (1 + 0.044715 * gv²)))
```

标准 tanh 近似注释（源码中亦有说明）：

```
gelu(gv) ≈ 0.5 * gv * (1 + tanh(√(2/π) * (gv + 0.044715 * gv³)))
```

### 4.2 GeGLU 逐元素乘

```
val = gelu(gv) * uv
```

### 4.3 静态 FP8 量化（E4M3）

```
inv_scale = 1 / max(act_scale_down, 1e-12)
hid_fp8 = clamp(val * inv_scale, -448, 448)  →  __nv_fp8_e4m3
```

核心 loop（`geglu_fp8_static_fp16_kernel` / `gate_silu_mul_merged_fp8_kernel`，逻辑相同）：

```cpp
float gv = merged[s * 2*H + h];
float uv = merged[s * 2*H + H + h];
float gelu = gv / (1.0f + expf(-1.5957691216057308f * gv * (1.0f + 0.044715f * gv * gv)));
float val = gelu * uv;
out[s*H + h] = quantize_e4m3(val, act_scale_down);
```

计算在 **fp32** 中完成（`__half2float` 提升），再 narrow 到 fp8。

---

## 5. 融合了什么

OpenPI 若分步实现：

```python
gate_lin = gate_proj(x_norm)    # C5 GEMM
up_lin   = up_proj(x_norm)      # C5 GEMM（FlashRT 已与 gate 合并）
hidden   = act_fn(gate_lin) * up_lin
hidden_fp8 = quantize(hidden)   # 供 down_proj
out      = down_proj(hidden_fp8)
```

FlashRT 合并 **GELU + mul + quant** 为 **1 个 elementwise kernel**：

```text
读 fg[S,2H] fp16 → GELU(gate) × up (fp32) → quant fp8 → 写 hid_fp8[S,H]
```

| 若拆开 | 问题 |
|--------|------|
| GELU kernel + mul kernel + `quantize_fp8_static_fp16` | 3 次 launch；`[S,H]` fp16 hidden 多次 DRAM 读写 |
| 融合为一步 | 1 次 launch；memory-bound 链合并（FlashRT 文档记 **5→1** 级融合收益） |

Launch：`<<<(S*H + 255)/256, 256>>>`，纯 elementwise，**memory-bound**。

---

## 6. 与 C5、down GEMM 的分工

这是 FFN 三段式中 **「非线性 + 量化」** 的必经节点：

```text
C5 GEMM (线性)          → fg fp16 [S, 2H]     # GELU **前**；不能在此 quant fp8
★ gate_geglu_merged    → hid_fp8 [S, H]      # GELU×up **后** quant（正确 quant 点）
C6 down GEMM (线性)    → fg fp16 [S, D]      # descale 用同一 act_scale_down
```

### 6.1 两个 scale 槽位（k=2 vs k=3）

| 槽位 k | scale | 标定对象 | 用于 |
|--------|-------|----------|------|
| 2 | `act_scale_gu` | post-attention norm 后激活 | C4→5 quant + C5 GEMM descale |
| **3** | **`act_scale_down`** | **GeGLU 输出** | **本 kernel quant + down GEMM descale** |

**`act_scale_down` 必须与下一行 down GEMM 传入的 `act_descale` 相同**（quant 除 `s_act`、GEMM epilogue 乘 `s_act` 配对）。

### 6.2 为何 C5 不直接出 fp8？

C5 输出是 **GELU 前的线性值**；本 kernel 的 fp8 是 **GELU×up 之后** 的值。非线性必须在 fp16/fp32 域上算，quant 点必须在 GeGLU 之后。详见 [`fp8_gemm_descale_fp16_c5.md`](fp8_gemm_descale_fp16_c5.md) §4 Q3。

---

## 7. OpenPI ↔ FlashRT 完整 FFN 对照

```python
# OpenPI（单层 Expert FFN）
x_norm, _ = post_attention_layernorm(x, cond=adarms_cond)
gate_lin = gate_proj(x_norm)
up_lin   = up_proj(x_norm)
hidden   = act_fn(gate_lin) * up_lin
out      = down_proj(hidden)
```

```text
# FlashRT
C4→5  xn_fp8 = quant(AdaRMSNorm(...))
C5    fg     = fp8_gemm(xn_fp8, gw)              # gate_lin | up_lin
★ C6  hid_fp8 = gate_geglu_merged(fg)            # hidden = GELU(gate)*up, quant
C6    fg     = fp8_gemm(hid_fp8, dw)             # out = down_proj(hidden)
C7    x     += fg * gate                         # 第二次 gated residual
```

---

## 8. 实现备注

| 项 | 说明 |
|----|------|
| Python 名 | `gate_geglu_merged_fp8_fp16`（语义：GeGLU，fp16 入，fp8 出） |
| 生产绑定 | `activation.cu::gate_silu_mul_merged_fp8_fp16`（历史命名含 silu） |
| decoder 副本 | `decoder_fused.cu::geglu_fp8_static_fp16`（同等数学） |
| 激活函数 | Pi0.5：**GELU tanh 近似**；非 SiLU/SwiGLU（Qwen3 等用 `silu_mul_split_fp8_fp16`） |
| 输入布局 | 要求 C5 合并 GEMM 的 `[gate \| up]` 交错 buffer，见 `weight_spec` CatGateUp |

---

## 9. 小结

| 问题 | 答案 |
|------|------|
| 做什么？ | `GELU(gate_linear) × up_linear`，再静态 quant 到 fp8 |
| denoise 对应？ | `layer.mlp()` 里 `act_fn(gate_proj(x)) * up_proj(x)` |
| 为何融合？ | GELU + mul + quant 是连续 elementwise 链，单 kernel 省 launch 与带宽 |
| 为何出 fp8？ | down GEMM 需要 FP8 激活；quant 必须在 **GELU 之后**（槽位 k=3） |
| 与 C5 分工？ | C5 线性 GEMM → fp16；本 kernel 非线性 + 量化 → `hid_fp8` → down GEMM |

**一句话**：本 kernel 是 FFN 的 **激活融合点**——读 C5 的 gate/up 线性 fp16 输出，做 GeGLU 非线性，按 down 路径标定 scale 量化成 fp8，衔接 C6 `down_proj` FP8 GEMM。
