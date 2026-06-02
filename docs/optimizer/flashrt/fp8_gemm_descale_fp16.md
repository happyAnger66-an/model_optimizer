# FP8 GEMM + Descale（`fp8_gemm_descale_fp16`）

> 实现：FlashRT `flash_rt_kernels*.so`（仓内 `flashrt_decoder/kernels.py` 加载为 `fvk`）  
> 编排：`pipeline.py::decoder_forward` 中 C2 / C4 / C5 / C6（down）  
> 权重 FP8：`weights.py::quant_fp8` + `w_scales`；激活 FP8：上游融合核 + `act_scales`  
> 对照：[`fusion_design.md`](fusion_design.md)（C1→C2 切口）、[`flashrt_denoise_fp8.md`](../ddup/flashrt_denoise_fp8.md)（标定）

---

## 1. 功能概述

`fp8_gemm_descale_fp16` 在 decoder 每层 FP8 路径上承担 **所有大矩阵乘**（QKV、O-proj、gate+up、down）：

- **输入**：激活 FP8（`uint8` / `float8_e4m3fn`）+ 权重 FP8 + 各自的 `act_scale`、`w_scale`
- **输出**：**fp16** 张量（直接写入 `qkv` / `fg` 等 buffer）
- **`descale`**：在 GEMM **epilogue** 里乘 `s_act × s_w`，**不是**先写出 fp8 结果再单独反量化

名字里的 **`fp16`** 表示 **输出 dtype**；**`fp8`** 表示 **A/B 矩阵为 FP8**。中间没有 `[S,N]` 的 fp8 GEMM 产物落盘。

---

## 2. 数据流（以 C2 QKV 为例）

```text
C1  fused_adarms_fp8_static_fp16
      x(fp16) + sa → xn_fp8          # 量化时用 act_scale_qkv (= s_act)

C2  fp8_gemm_descale_fp16
      xn_fp8 × qw_fp8  ──Tensor Core──►  epilogue × (s_act × s_w)  ──►  qkv(fp16)

C2b qkv_split_rope_kvcache_fp16(qkv, …)   # 后续在 fp16 上 RoPE / 写 KV
```

**常见误解**：「GEMM 输出 fp8，再 dequant 成 fp16」。  
**实际**：累加器多为 fp32（实现细节），epilogue 一次乘 scale 后 **直接写 fp16 `qkv`**；`DecoderBuffers` 无 `qkv_fp8`。

---

## 3. 数学语义

### 3.1 离线权重量化（`quant_fp8`）

\[
W_{\text{fp8}} \approx W / s_w,\quad s_w = \max(|W|)/448
\]

每层、每个量化点一个 `w_scale`（per-tensor），存在 `w_scales[l*4+k]`。

### 3.2 激活量化（C1 / `quantize_fp8_static_fp16` 等）

\[
X_{\text{fp8}} \approx X_{\text{norm}} / s_{\text{act}}
\]

`s_act` 来自标定 `act_scales[l*4+k]`，**必须与下游 GEMM 传入的 `act_scale` 指针相同**。

### 3.3 GEMM + descale

希望近似 OpenPI 的 fp16 线性层：

\[
Y = X_{\text{norm}} W^\top \approx (X_{\text{fp8}} W_{\text{fp8}}^\top)\cdot s_{\text{act}}\cdot s_w
\]

Kernel 用 Tensor Core 算 \(X_{\text{fp8}} W_{\text{fp8}}^\top\)（e4m3），在 epilogue 乘 \(s_{\text{act}} s_w\)，输出 **fp16** \(Y\)。

---

## 4. C2：合并 QKV（与 OpenPI 对照）

### 4.1 调用形式

```python
fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                          act_scale_qkv, w_scale_qkv, stream)
```

| 参数 | 含义 |
|------|------|
| `xn_fp8` | C1 输出的激活，`[S*D]` FP8 |
| `qw_ptr` | 第 `l` 层合并 Q+K+V 权重，`[D, 2560]` FP8（`weights.py` repack） |
| `qkv` | 输出 **fp16**，`[S, 2560]` = Q(`NH*HD`) + K(`HD`) + V(`HD`) |
| `S, 2560, D` | GEMM 形状 `M=S, N=2560, K=D`（`C[M,N] = A[M,K] @ B[K,N]`） |
| `act_scale_qkv` | 与 C1 **同一** 槽位 `l*4+0` |
| `w_scale_qkv` | 该层 QKV 权重 scale |

### 4.2 OpenPI 等价

| OpenPI（`gemma_pytorch.py`） | FlashRT C2 |
|------------------------------|------------|
| `q_proj(x_norm)`、`k_proj`、`v_proj` 三次 `nn.Linear` | **一次** `qw` GEMM |
| 输入 fp16 `hidden_states` | 输入 **FP8** `xn_fp8`（Norm 后在 C1 已量化） |
| 输出 fp16 Q/K/V | 输出 fp16 `qkv`，C2b 再 split + RoPE |

---

## 5. 管线内其它 `fp8_gemm_descale_fp16` 调用

每层 4 个 FP8 GEMM，与 [`flashrt_denoise_fp8.md`](../ddup/flashrt_denoise_fp8.md) §2.1 一致：

| 槽位 k | 阶段 | 激活输入 | 权重 | 输出 buffer | 形状 (M,N,K) |
|--------|------|----------|------|-------------|--------------|
| 0 | C2 QKV | `xn_fp8`（C1） | `qw` | `qkv` | S, 2560, D |
| 1 | C4 O | `ctx_fp8`（`quantize_fp8_static_fp16`） | `ow` | `fg` | S, D, NH*HD |
| 2 | C5 gate+up | `xn_fp8`（C4→5 `gate_res_adarms`） | `gw` | `fg` | S, 2H, D |
| 3 | C6 down | `hid_fp8`（`gate_geglu_merged`） | `dw` | `fg` | S, D, H |

共性：**上游量化点的 `act_scale` 与本次 GEMM 的 `act_scale` 参数必须一致**；输出均为 fp16，供后续 fp16 kernel（Attention、RoPE、GeGLU 等）或下一量化点使用。

---

## 6. 与 C1 / 算子命名的关系

- **C1** `fused_adarms_fp8_static_fp16`：名字里的 `fp16` 指残差流 `x`、`gate`、`sa` 接口；**GEMM 支路**输出是 `xn_fp8`（见 [`fusion_design.md`](fusion_design.md) §4）。
- **C2** `fp8_gemm_descale_fp16`：消费 `xn_fp8`，**不**读 fp16 的 norm 结果；**不**产生 fp8 的 `qkv`。

---

## 7. 相关文件

| 文件 | 角色 |
|------|------|
| `pipeline.py` | 四处 `fp8_gemm_descale_fp16` 调用与注释 |
| `weights.py` | `quant_fp8`、`dec_qkv_flat` / `ae_w_scales` repack |
| `driver.py` | `qkv`(fp16)、`xn_fp8`、`ctx_fp8`、`hid_fp8` buffer |
| `fusion_design.md` | 融合边界与 C1→C2 切口 |
| `flashrt_denoise_fp8.md` | 激活标定与 `l*4+k` 槽位 |

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-02 | 初版：descale 语义、C2 QKV、非「fp8 输出再 dequant」、四层 GEMM 表 |
