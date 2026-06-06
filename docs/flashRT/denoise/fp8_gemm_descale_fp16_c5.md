# C5：`fp8_gemm_descale_fp16`（Gate+Up 合并 GEMM）说明

> **实现**：`third_party/FlashRT/csrc/kernels/decoder_fused.cu::fp8_gemm_descale_fp16`（cuBLASLt host wrapper，非自定义 CUDA kernel）  
> **Python 绑定**：`fvk.fp8_gemm_descale_fp16`  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` — C5（每层 FFN 第一段）  
> **量化槽位**：`k=2`（`act_scales[l*4+2]` / `w_scales[l*4+2]`）  
> **上游**：[`gate_res_adarms_fp8_static_fp16.md`](gate_res_adarms_fp8_static_fp16.md)（C4→5，产出 `xn_fp8`）  
> **下游**：C6 `gate_geglu_merged_fp8_fp16` + down GEMM（[`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md) §5 槽位 k=3）  
> **通用 GEMM 语义**：[`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md)

---

## 1. 功能概述

C5 是 Pi0.5 denoise **FFN 的第一段大矩阵乘**：用 post-attention 归一化后的 **FP8 激活** `xn_fp8`，一次 Tensor Core GEMM 同时完成 OpenPI 里的 **`gate_proj` + `up_proj`** 两次 Linear，输出写入 fp16 buffer `fg`。

pipeline 调用：

```python
act_scale_gu = act_scales + (l * 4 + 2) * 4
w_scale_gu   = w_scales   + (l * 4 + 2) * 4
gw_ptr       = gw + l * D * H * 2

fvk.fp8_gemm_descale_fp16(
    xn_fp8, gw_ptr, fg,
    S, H * 2, D,              # M, N, K
    act_scale_gu, w_scale_gu, stream,
)
```

OpenPI 等价（`GemmaMLP.forward`，`modeling_gemma.py`）：

```python
def forward(self, x):
    down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

FlashRT 将 `gate_proj(x)` 与 `up_proj(x)` **合并为一次 GEMM**；`act_fn`（GELU tanh 近似）× `up` 留给 C6 融合核处理。

---

## 2. 在 Denoise 流水线中的位置

```text
扩散步 s:
  for layer l in 0..17:
    ...
    C4   quantize_fp8 + fp8_gemm_descale_fp16 (k=1) → fg        # o_proj
    C4→5 gate_res_adarms_fp8_static_fp16           → xn_fp8    # post_attention_layernorm
    ★ C5 fp8_gemm_descale_fp16 (k=2)               → fg        # gate_proj + up_proj
    C6   gate_geglu_merged_fp8_fp16               → hid_fp8   # GELU(gate) × up
    C6   fp8_gemm_descale_fp16 (k=3)               → fg        # down_proj
    C7   gated residual + 下一层 norm / 最后一层 gate_res_fp16
```

对应 `denoise_step` → `paligemma_with_expert.forward(..., suffix_embs)` 中 **Expert 第 `l` 层 `layer.mlp()` 的前半段**（两次 Linear，尚未做激活函数）。

`gemma_pytorch.py` 侧等价片段：

```python
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
out_emb = layer.mlp(out_emb)
# mlp 内部: gate_proj(x) 与 up_proj(x) → act_fn(gate) * up → down_proj
```

| OpenPI | FlashRT |
|--------|---------|
| `post_attention_layernorm` | C4→5 `gate_res_adarms_fp8_static_fp16` |
| `gate_proj(x)` + `up_proj(x)` | **C5 合并 GEMM → `fg[S, 2H]`** |
| `act_fn(gate) * up` | C6 `gate_geglu_merged_fp8_fp16` |
| `down_proj(...)` | C6 down GEMM（k=3） |

---

## 3. 计算内容

### 3.1 数学语义

希望近似 OpenPI fp16 线性层：

```
Y = X_norm @ W_gu^T
  ≈ (X_fp8 @ W_gu_fp8^T) × s_act × s_w
```

- **输入** `xn_fp8`：`[S, D]` FP8，C4→5 产出（post-attention AdaRMSNorm 后量化）
- **权重** `gw` / `dec_gu_flat`：`[D, 2H]` FP8，离线 repack 的 gate+up 合并权重
- **输出** `fg`：`[S, 2H]` **fp16**，布局 `[gate_linear \| up_linear]`（前 H 列 gate 分支，后 H 列 up 分支）
- **GEMM 形状** `(M, N, K) = (S, 2H, D)`

合并权重动机：OpenPI 两次 `[S,D]→[S,H]` 的 Linear 共享同一输入 `x`，合并为 `[S,D]→[S,2H]` 只需 **一次** GEMM launch，权重在 `weights.py` 中 cat 为 `dec_gu_flat`。

### 3.2 参数对照

| 参数 | 形状 / 类型 | 含义 |
|------|-------------|------|
| `xn_fp8` | `[S, D]` fp8 | 激活 A |
| `gw_ptr` | `[D, 2H]` fp8 | 第 `l` 层权重 B，`gw + l * D * H * 2` |
| `fg` | `[S, 2H]` fp16 | 输出 C |
| `act_scale_gu` | device fp32 | `act_scales[l*4+2]`，**须与 C4→5 quant 相同** |
| `w_scale_gu` | device fp32 | `w_scales[l*4+2]`，gate+up 权重量化 scale |

### 3.3 实现路径（cuBLASLt）

与 C2/C4/C6 共用同一 host 函数 `fp8_gemm_descale_fp16`：

1. FP8×FP8 Tensor Core 点积，**fp32 累加器**
2. Epilogue 乘 **`s_act × s_w`**（descale）
3. **直接写 fp16** 到 `fg`，无 `[S, 2H]` fp8 中间 GEMM 产物

详见 [`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md) §3–§5。

### 3.4 下游 C6 如何使用 `fg`

C6 **不会**直接把 `fg` 送进 down GEMM，而是先做 GeGLU 非线性：

```text
fg[s, 0:H]   → gate 线性输出（GELU 前）
fg[s, H:2H]  → up 线性输出
C6: GELU(gate) × up → hid_fp8 [S, H] → down GEMM (k=3)
```

C6 kernel（`geglu_fp8_static_fp16_kernel`）在 fp32 中对 fp16 的 gate/up 做 GELU tanh 近似与逐元素乘，再按 **`act_scale_down`（`l*4+3`）** 量化到 `hid_fp8`。

---

## 4. 常见问题：descale 是什么？

### Q1：epilogue 乘 scale 后写 fp16，scale 就是 FP8 反量化 scale 吗？

**是。** epilogue 里的 scale 就是 FP8 **per-tensor 静态 descale 系数**（激活一个、权重一个），不是逐元素类型转换。

离线量化：

```
W_fp8  ≈ W / s_w        （s_w = max(|W|) / 448）
X_fp8  ≈ X / s_act      （s_act = act_scales[l*4+k]，标定得到）
```

运行时 GEMM epilogue：

```
Y_fp16 ≈ (X_fp8 @ W_fp8^T) × s_act × s_w
```

| 指针 | 含义 |
|------|------|
| `act_descale` → `s_act` | 激活反量化 scale；**与上游 quant 用的是同一个 device 指针** |
| `w_descale` → `s_w` | 权重反量化 scale |

C5 约束：`act_scale_gu` 必须同时用于：
- C4→5 `gate_res_adarms_fp8_static_fp16` 内的 quant（`× 1/s_act`）
- C5 GEMM epilogue descale（`× s_act`）

否则 `Y_fp16` 数值域错误，无法近似 OpenPI 的 `nn.Linear` 输出。

### Q2：descale 是「反量化成 fp16」吗？

**语义上接近，实现上是一步融合，不是两步。**

| 概念上（分步） | FlashRT 实际 |
|----------------|--------------|
| `X_fp16 ≈ X_fp8 × s_act` | 不做 |
| `W_fp16 ≈ W_fp8 × s_w` | 不做 |
| 再 fp16 GEMM | FP8×FP8 GEMM + epilogue **一次** `× s_act × s_w` → 直接写 fp16 |

目的：让 `Y_fp16` 近似 OpenPI fp16 `Linear` 输出，供后续 **非线性算子**（此处为 GELU×mul）使用；不在 DRAM 中落 fp16 的 X/W 中间张量。

### Q3：为什么不直接出 fp8？反正下游 C6 也要 fp8？

**C5 的 fp8 与 C6 需要的 fp8 不是同一语义，量化点也不同。**

#### 原因 1：非线性必须在 fp16（或更高精度）域上执行

OpenPI FFN：

```python
act_fn(gate_proj(x)) * up_proj(x)
#      ^^^^^^^^^^^^^^^^
#      非线性作用在线性 GEMM 输出上，不是 quant 后的 fp8 上
```

数据流：

```text
C5 GEMM  → fg fp16   # gate/up 的 **线性** 输出（GELU 前）
C6 GeGLU → hid_fp8   # GELU(gate) × up 之后 **再** quant
C6 GEMM  → down
```

若 C5 直接出 fp8：
- **方案 A**：对线性输出先 quant → 在 fp8 上做 GELU → 精度差、语义错（非线性应在反量化后的实值域上算）
- **方案 B**：C6 先 dequant fp8→fp16 再 GELU → 多一次往返，无收益

**正确 quant 点**是 GeGLU **之后**、down GEMM **之前**——由 C6 融合核 `gate_geglu_merged_fp8_fp16` 完成。

#### 原因 2：两个 scale 槽位标定对象不同

| 槽位 k | scale | 标定 / 用于 |
|--------|-------|-------------|
| **2** | `act_scale_gu` | post-attention norm **输入** C5 GEMM 的激活 |
| **3** | `act_scale_down` | **GELU(gate)×up 之后**、down GEMM 输入 |

当前 pipeline **没有**为「gate/up 线性 GEMM 输出」单独标定 fp8 scale；down 路径的 fp8 scale 针对的是 **GeGLU 输出**。

#### 原因 3：decoder 四层 GEMM 的统一设计

四层 FP8 GEMM 均为 **FP8 输入 → fp16 线性输出 → 下游 fp16 算子或融合 quant**：

| k | 阶段 | GEMM 输出 fp16 供… |
|---|------|-------------------|
| 0 | QKV | RoPE / Attention |
| 1 | O-proj | gated residual |
| 2 | gate+up | GELU × up（C6） |
| 3 | down | gated residual（C7） |

**进入 GEMM 的激活** 用 fp8；**GEMM 线性输出** 保持 fp16，直到下一个业务正确的 quant 点。

#### 若强行 C5→fp8 的对比

```text
当前（推荐）:
  GEMM → fp16 linear → GELU×up (fp32) → quant fp8 → down GEMM

强行 C5→fp8:
  GEMM → fp8 linear → ??? GELU → fp8 → down GEMM
  （仍需 dequant 或接受精度损失；scale 需重新标定）
```

---

## 5. 与 C4→5、C6 的数据流

```text
                    act_scale_gu (k=2)
                           │
C4→5 gate_res_adarms ──► xn_fp8 ──► C5 fp8_gemm_descale_fp16 ──► fg [S,2H] fp16
                           │              │ w_scale_gu (k=2)
                           │              │ epilogue: × s_act × s_w
                           │              ▼
                           │         gate_linear | up_linear
                           │              │
                           │              ▼  act_scale_down (k=3)
                           │         C6 gate_geglu_merged
                           │              ▼
                           └── (同一 s_act) hid_fp8 ──► C6 down GEMM (k=3)
```

要点：
- C4→5 与 C5 GEMM **共享** `act_scale_gu`（quant 与 descale 配对）
- C6 GeGLU 使用 **独立的** `act_scale_down`（k=3）
- `fg` 在 C5→C6 之间是 **fp16 线性中间结果**，不是多余的 dtype 转换

---

## 6. OpenPI ↔ FlashRT 完整 FFN 对照

```python
# OpenPI（单层 Expert FFN）
x_norm, gate = post_attention_layernorm(x, cond=adarms_cond)
gate_lin = gate_proj(x_norm)          # [S, H]
up_lin   = up_proj(x_norm)            # [S, H]
hidden   = act_fn(gate_lin) * up_lin  # [S, H]
out      = down_proj(hidden)          # [S, D]
```

```text
# FlashRT
C4→5  xn_fp8 = quant(AdaRMSNorm(...))           # 对应 x_norm
C5    fg = fp8_gemm(xn_fp8, gw)                 # 对应 gate_lin | up_lin 合并
C6    hid_fp8 = quant(GELU(fg[:H]) * fg[H:])    # 对应 hidden
C6    fg = fp8_gemm(hid_fp8, dw)                 # 对应 out = down_proj(hidden)
C7    x += fg * gate                             # 第二次 gated residual
```

---

## 7. 小结

| 问题 | 答案 |
|------|------|
| C5 做什么？ | gate+up 合并 FP8 GEMM，近似 `gate_proj` + `up_proj` |
| denoise 对应？ | `layer.mlp()` 前半，post-attention norm 之后、GELU×up 之前 |
| descale 的 scale？ | 就是 FP8 反量化系数 `s_act`、`s_w`；epilogue 一次乘积后直接写 fp16 |
| 为何不出 fp8？ | GEMM 输出是 **GELU 前的线性值**；下游 fp8 应对 **GeGLU 之后** 的激活（C6，槽位 k=3） |
| 谁产出下游 fp8？ | C6 `gate_geglu_merged_fp8_fp16` 融合 GELU×mul + quant |

**一句话**：C5 用 FP8 Tensor Core 做合并 Linear，descale 后在 epilogue 还原为 fp16 线性输出；fp16 是 GELU 非线性的必要输入，真正的 down-GEMM 用 fp8 由 C6 在正确 quant 点产出。
