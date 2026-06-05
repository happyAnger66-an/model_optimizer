# C2/C4/C5/C6：`fp8_gemm_descale_fp16` 算子说明

> **实现**：`third_party/FlashRT/csrc/kernels/decoder_fused.cu::fp8_gemm_descale_fp16`  
> **底层**：cuBLASLt FP8 Tensor Core GEMM（**非**自定义 CUDA kernel）  
> **Python 绑定**：`fvk.fp8_gemm_descale_fp16`（`flashrt_decoder/kernels.py` 加载）  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` — C2 / C4 / C5 / C6（down）  
> **上游融合核**：[`fused_adarms_fp8_static_fp16.md`](fused_adarms_fp8_static_fp16.md)（C1→C2 切口）  
> **扩展文档**：[`docs/optimizer/flashrt/fp8_gemm_descale_fp16.md`](../../optimizer/flashrt/fp8_gemm_descale_fp16.md)

---

## 1. 功能概述

`fp8_gemm_descale_fp16` 是 Pi0.5 denoise **FP8 路径上所有大矩阵乘** 的统一入口：

- **输入**：激活 FP8 + 权重 FP8 + 各自的 `act_scale`、`w_scale`（device 端 float 指针）
- **输出**：**fp16** 张量（直接写入 `qkv` / `fg` 等 buffer）
- **descale**：在 GEMM **epilogue** 里乘 `s_act × s_w`，**不是**先写出 fp8 结果再单独反量化

名字里的 **`fp8`** 表示 A/B 矩阵为 FP8；**`fp16`** 表示 **输出 dtype**。中间不存在 `[M, N]` 的 fp8 GEMM 产物落盘。

OpenPI 等价物：`nn.Linear`（`q/k/v_proj`、`o_proj`、`gate_proj/up_proj/down_proj`）。FlashRT 将 Q+K+V 合并为一次 `qw` GEMM，四层 decoder 内共 **4 次** 调用本函数。

---

## 2. 在 Denoise 流水线中的位置

每个扩散步 `s` 执行 18 层 Gemma Expert，每层 4 个 FP8 量化槽位 `l*4+k`（`k=0..3`）：

```text
扩散步 s:
  for layer l in 0..17:
    C1   fused_adarms_fp8_static_fp16        → xn_fp8
    C2   fp8_gemm_descale_fp16  (k=0)      → qkv(fp16)     # QKV
    C2b  qkv_split_rope_kvcache_fp16
    C3   Attention / FMHA
    C4   quantize_fp8 + fp8_gemm_descale_fp16 (k=1) → fg   # O-proj
    C4→5 gate_res_adarms_fp8_static_fp16   → xn_fp8
    C5   fp8_gemm_descale_fp16  (k=2)      → fg(fp16)     # gate+up 合并
    C6   gate_geglu_merged + fp8_gemm_descale_fp16 (k=3) → fg  # down
    C7   gated residual → 下一层 C1
```

### 2.1 四层 GEMM 对照表

| 槽位 k | 阶段 | 激活输入 | 权重 | 输出 buffer | 形状 (M, N, K) |
|--------|------|----------|------|-------------|----------------|
| 0 | C2 QKV | `xn_fp8`（C1） | `qw` | `qkv` | S, 2560, D |
| 1 | C4 O | `ctx_fp8`（`quantize_fp8_static_fp16`） | `ow` | `fg` | S, D, NH×HD |
| 2 | C5 gate+up | `xn_fp8`（C4→5） | `gw` | `fg` | S, 2H, D |
| 3 | C6 down | `hid_fp8`（GeGLU） | `dw` | `fg` | S, D, H |

共性：

- 上游量化点的 `act_scale` 与本次 GEMM 的 `act_descale` **必须相同**
- 输出均为 fp16，供 RoPE、Attention、GeGLU 等后续 fp16 kernel 使用

### 2.2 OpenPI ↔ FlashRT

| OpenPI（pi05） | FlashRT |
|----------------|---------|
| `q_proj` / `k_proj` / `v_proj` 三次 `nn.Linear` | **一次** `qw` `[D, 2560]` GEMM（C2） |
| `o_proj` | C4 GEMM |
| `gate_proj` + `up_proj` | C5 合并 `gw` GEMM |
| `down_proj` | C6 GEMM |
| 输入 fp16 激活 | 输入 **FP8** 激活（上游融合核已量化） |
| 输出 fp16 | 输出 fp16（descale 在 epilogue，非后处理） |

---

## 3. 数学语义

### 3.1 离线权重量化（`weights.py::quant_fp8`）

```
W_fp8 ≈ W / s_w,    s_w = max(|W|) / 448
```

每层、每个量化点一个 `w_scale`（per-tensor），存在 `w_scales[l*4+k]`。

### 3.2 激活量化（C1 / `quantize_fp8_static_fp16` 等）

```
X_fp8 ≈ X_norm / s_act
```

`s_act` 来自标定 `act_scales[l*4+k]`，**必须与下游 GEMM 传入的 `act_descale` 指针相同**。

### 3.3 GEMM + descale（运行时）

希望近似 OpenPI 的 fp16 线性层：

```
Y = X_norm @ W^T  ≈  (X_fp8 @ W_fp8^T) × s_act × s_w
```

- Tensor Core 在 **fp32 累加器** 中计算 FP8×FP8 点积
- Epilogue 乘 `s_act × s_w`，**直接写 fp16** `Y`
- 无 `[M, N]` fp8 中间 buffer

### 3.4 常见误解

| 误解 | 实际 |
|------|------|
| GEMM 输出 fp8，再 dequant 成 fp16 | epilogue 一次乘 scale 后直接写 fp16 |
| C2 读 fp16 norm 结果 | 只读 C1 产出的 `xn_fp8` |
| `DecoderBuffers` 有 `qkv_fp8` | 无；`qkv` 为 fp16 |

---

## 4. 参数与张量布局

### 4.1 函数签名

```python
fvk.fp8_gemm_descale_fp16(A_fp8, B_fp8, C_fp16, M, N, K, act_descale, w_descale, stream)
```

| 参数 | 含义 | C2 QKV 示例 |
|------|------|-------------|
| `A_fp8` | 激活，逻辑 row-major `[M, K]` | `xn_fp8` `[S, D]` |
| `B_fp8` | 权重，逻辑 row-major `[K, N]` | `qw` `[D, 2560]` |
| `C_fp16` | 输出，逻辑 row-major `[M, N]` | `qkv` `[S, 2560]` |
| `M, N, K` | GEMM 形状 `C = A @ B` | `S, 2560, D` |
| `act_descale` | device 指针 → `s_act` | `act_scales + (l*4+0)*4` |
| `w_descale` | device 指针 → `s_w` | `w_scales + (l*4+0)*4` |

### 4.2 C2 调用示例

```python
act_scale_qkv = act_scales + (l * 4 + 0) * 4
w_scale_qkv   = w_scales + (l * 4 + 0) * 4
qw_ptr        = qw + l * D * 2560
fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                          act_scale_qkv, w_scale_qkv, stream)
```

`act_scale_qkv` 须与 C1 `fused_adarms_fp8_static_fp16` 使用的 descale **相同**。

---

## 5. 代码详细流程（Step 0 → 3）

本函数是 **cuBLASLt host wrapper**，核心逻辑分四步：

```text
Step 0  懒初始化 cuBLASLt handle + 32MB workspace
    ↓
Step 1  按 (M,N,K) 查找/构建缓存的 MatmulDesc + Layout + Algo
    ↓
Step 2  绑定本层 descale 指针（act_scale, w_scale）
    ↓
Step 3  cublasLtMatmul 启动 FP8 Tensor Core GEMM → fp16 输出
```

### 5.1 Step 0：一次性全局初始化

```cpp
if (!g_fp8_lt) {
    cublasLtCreate(&g_fp8_lt);
    cudaMalloc(&g_fp8_ws, 32 * 1024 * 1024);  // 32 MB
}
```

- 首次调用时创建 cuBLASLt handle
- 分配 32MB workspace，供 `MatmulAlgoGetHeuristic` 选择最优 Tensor Core kernel（与 pi05 生产环境一致）

### 5.2 Step 1：Descriptor + Algo 缓存

Pi0.5 denoise 的 `(M, N, K)` 在推理时 **固定**，Descriptor 创建和 algo heuristic 开销大，因此按 `(M, N, K)` 缓存：

| 缓存组件 | 作用 |
|----------|------|
| `LtGemmKey{M,N,K}` | `g_lt_cache` 哈希键 |
| `MatmulDesc` | `CUBLAS_COMPUTE_32F` 累加器；`TRANSA=TRANSB=N` |
| `Adesc` | 权重 layout：col-major `(N, K)` E4M3 |
| `Bdesc` | 激活 layout：col-major `(K, M)` E4M3 |
| `Cdesc` | 输出 layout：col-major `(N, M)` FP16 |
| `algo` | heuristic 选出的最优 FP8 kernel |

**首次** 遇到新 `(M,N,K)` 时：

1. `cublasLtMatmulDescCreate` — fp32 累加，epilogue 写 fp16
2. `cublasLtMatrixLayoutCreate` × 3 — 见 §6 布局说明
3. `cublasLtMatmulAlgoGetHeuristic` — 在 32MB workspace 约束下选 algo
4. 存入 `g_lt_cache`，后续同形状调用 **O(1) 命中**

### 5.3 Step 2：绑定 per-call descale 指针

```cpp
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, ...);
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, ...);
```

- 指针指向 **device 端 float**（非 host 值）
- **每层、每槽位不同**，每次调用更新；Descriptor/Algo 不变
- cuBLAS 矩阵 A = 权重 → `A_SCALE → w_descale`
- cuBLAS 矩阵 B = 激活 → `B_SCALE → act_descale`

### 5.4 Step 3：启动 GEMM

```cpp
float alpha = 1.0f, beta = 0.0f;
cublasLtMatmul(lt, desc, &alpha,
    B_fp8, Adesc,    // cuBLAS "A" = 权重
    A_fp8, Bdesc,    // cuBLAS "B" = 激活
    &beta, C_fp16, Cdesc, C_fp16, Cdesc,
    &algo, workspace, workspace_sz, stream);
```

- `C = alpha * A * B + beta * C`，`beta=0` 覆盖输出
- Tensor Core 执行 FP8×FP8 矩阵乘
- Epilogue 乘 `s_act × s_w`，写 fp16 `C`

---

## 6. cuBLASLt 布局映射

cuBLAS 使用 **列主序（column-major）**，与 Python/C 习惯的行主序不同。逻辑上的 row-major GEMM：

```
C_row[M, N] = A_act_row[M, K] @ B_w_row[K, N]
```

映射为 cuBLASLt 列主序：

| cuBLAS 矩阵 | 物理指针 | Layout | 含义 |
|-------------|----------|--------|------|
| A（权重） | `B_fp8` | `(N, K)` col-major, lda=N | 逻辑 `B_w[K,N]` |
| B（激活） | `A_fp8` | `(K, M)` col-major, lda=K | 逻辑 `A_act[M,K]` |
| C（输出） | `C_fp16` | `(N, M)` col-major, lda=N | 逻辑 `C[M,N]` |

等价关系：

```
C_col(N, M) = A_col(N, K) × B_col(K, M)
            ≡ C_row(M, N) = A_row(M, K) @ B_row(K, N)
```

**注意** `cublasLtMatmul` 调用时指针顺序与参数名相反：第一个矩阵指针是 `B_fp8`（权重），第二个是 `A_fp8`（激活）。

---

## 7. C2 QKV 完整数据流

```text
C1  fused_adarms_fp8_static_fp16
      x(fp16) + sa → xn_fp8              # 量化: xn ≈ x_norm / s_act

C2  fp8_gemm_descale_fp16(xn_fp8, qw, qkv, S, 2560, D, s_act, s_w)
      Tensor Core:  xn_fp8 @ qw_fp8
      epilogue:     × s_act × s_w
      → qkv(fp16)                        # 等价 OpenPI q/k/v_proj(x_norm)

C2b qkv_split_rope_kvcache_fp16(qkv, ...)   # fp16 上 split + RoPE + 写 KV
```

OpenPI 三次独立 Linear：

```python
q = q_proj(x_norm)   # [S, NH*HD]
k = k_proj(x_norm)   # [S, HD]
v = v_proj(x_norm)   # [S, HD]
```

FlashRT 合并权重为 `qw [D, 2560]`，一次 GEMM 输出 `[S, 2560]` 的 `qkv`，C2b 再 split。

---

## 8. 与 C1 的关系

| 算子 | 名字中 `fp16` 指 | GEMM 支路输出 |
|------|------------------|---------------|
| C1 `fused_adarms_fp8_static_fp16` | 残差流 `x`、`gate`、`sa` 接口 | **`xn_fp8`**（fp8） |
| C2 `fp8_gemm_descale_fp16` | **输出** dtype | **`qkv`**（fp16） |

融合边界设计（见 [`fusion_design.md`](../../optimizer/flashrt/fusion_design.md)）：

- C1 在 GEMM **入口前** 融合 Norm + 量化 → 避免 `[S,D]` fp16 norm 落盘
- C2 消费 `xn_fp8`，不读 fp16 norm 结果
- `act_scale_qkv` 在 C1 量化与 C2 descale 间 **共用**

---

## 9. 缓存与性能要点

| 设计 | 原因 |
|------|------|
| `(M,N,K)` 级 Descriptor/Algo 缓存 | 创建 + heuristic 昂贵；denoise 形状固定，一次构建全程复用 |
| 32MB workspace | 允许 cuBLASLt 选择需要较大临时空间的快速 algo |
| per-call 更新 scale 指针 | 18 层 × 4 槽位 scale 不同，但 shape 相同 |
| FP8 Tensor Core + fp32 累加 | Blackwell/Ampere+ 原生 FP8 GEMM；epilogue descale 无额外 kernel |

---

## 10. 相关文件

| 文件 | 角色 |
|------|------|
| `third_party/FlashRT/csrc/kernels/decoder_fused.cu` | `fp8_gemm_descale_fp16` 实现 |
| `docs/flashRT/decoder_fused.cu` | 带详细注释的可读副本 |
| `third_party/FlashRT/csrc/bindings.cpp` | pybind11 导出 |
| `src/model_optimizer/infer/native/flashrt_decoder/pipeline.py` | 四层 GEMM 调用编排 |
| `src/model_optimizer/infer/native/flashrt_decoder/driver.py` | `qkv`(fp16)、`xn_fp8`、`ctx_fp8`、`hid_fp8` buffer |
| `src/model_optimizer/infer/native/flashrt_decoder/weights.py` | `quant_fp8`、`qw/ow/gw/dw` repack |
| `docs/flashRT/denoise/fused_adarms_fp8_static_fp16.md` | C1 上游融合核 |
| `docs/optimizer/flashrt/fusion_design.md` | 融合边界与 C1→C2 切口 |
| `docs/optimizer/flashrt/fp8_gemm_descale_fp16.md` | 精简版技术说明 |
