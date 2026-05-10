# `GemmRunner::fp8_nn_bias`：FP8 矩阵乘 + Bias Epilogue（FP16 输出）

本文说明 FlashRT 中 **`fp8_nn_bias`** 的数学含义、在 **SigLIP QKV 投影** 中的用法，以及在 **`gemm_runner.cu`** 里基于 **cuBLASLt** 的实现流程。

---

## 1. 在 SigLIP 里做什么（`shared_primitives.siglip_forward`）

在每一层 attention 前，已用 `layer_norm_fp8` 得到激活 **`x_fp8`**（形状 **`[S, D]`**，FP8 E4M3）。随后 QKV 线性层为：

```text
qkv[S, 3D] = alpha * ( x_fp8[S, D] @ W_qkv[D, 3D] ) + bias_qkv[3D]
```

对应 Python 调用（`M=S`, `N=3*D`, `K=D`，`alpha` 为该层标量 `a_qkv`）：

```python
gemm.fp8_nn_bias(
    x_fp8, weights['qkv_w'][l], qkv, weights['qkv_b'][l],
    S, 3 * D, D, a_qkv, stream,
)
```

- **`A`**（`x_fp8`）：激活，逻辑 **`[M, K]`** = **`[S, D]`**，FP8。  
- **`B`**（`qkv_w`）：权重，逻辑 **`[K, N]`** = **`[D, 3D]`**，FP8（与 `bf16_nn` / pi05 约定一致：**行主序、不转置**）。  
- **`D`**（`qkv`）：输出 **FP16**，逻辑 **`[M, N]`** = **`[S, 3D]`**，供后续 FMHA 将 **Q/K/V** 视为同一大缓冲里的三段（stride `3*D`）。

---

## 2. 数学定义（与 C++ 注释一致）

对 **`float` 标量 `alpha`**、**bias 长度 `N`**：

\[
D_{ij} = \alpha \cdot \sum_{t=0}^{K-1} A_{it}\, B_{tj} + \mathrm{bias}_j
\quad,\quad
i \in [0,M),\; j \in [0,N)
\]

- **`A`**：`M×K`，`CUDA_R_8F_E4M3`  
- **`B`**：`K×N`，`CUDA_R_8F_E4M3`  
- **`D`**：`M×N`，`CUDA_R_16F`（**FP16 累加/写出**）  
- **Bias**：cuBLASLt **`CUBLASLT_EPILOGUE_BIAS`**，按列 **`j`** 加在输出上（与线性层 `+ bias` 一致）。

**与 `fp8_nn_dev` 的区别**：`fp8_nn_dev` 使用 **`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `B_SCALE_POINTER`**（设备上每侧 descale）。**`fp8_nn_bias` 在缓存描述符里不设 A/B scale 指针**；量化/反量化相关的 **单层标量** 由调用方传入的 **`alpha`** 作为 **`cublasLtMatmul` 的标量 `alpha`** 进入计算（与 pi05 侧「单层一个 scale」的用法一致）。SigLIP 里该 `alpha` 来自 **`weights['alpha']`** 中按层索引的 **`a_qkv`**。

---

## 3. 实现流程（`gemm_runner.cu`）

实现位置：`third_party/FlashRT/csrc/gemm/gemm_runner.cu`（约 1040–1076 行）。

### 3.1 描述符与缓存

1. **`GemmKey`**：`{100, M, N + 2000000, K}` — 在同类 FP8 变体里区分 **`fp8_nn_bias`**（与注释「pi05 pattern」一致）。  
2. **缓存未命中时** 创建并缓存：  
   - **`matmul_desc`**：`CUBLAS_COMPUTE_32F`，输出累加类型 **`CUDA_R_32F`**；**`TRANSA` / `TRANSB` 均为 `N`**。  
   - **Epilogue**：**`CUBLASLT_EPILOGUE_BIAS`**。  
   - **Layout**（与内部 `cublasLtMatmul` 的矩阵顺序一致，见下一小节）：  
     - `A_desc`：`CUDA_R_8F_E4M3`，**`rows=N, cols=K, ld=N`**  
     - `B_desc`：`CUDA_R_8F_E4M3`，**`rows=K, cols=M, ld=K`**  
     - `D_desc`：`CUDA_R_16F`，**`rows=N, cols=M, ld=N`**  
   - **`cublasLtMatmulAlgoGetHeuristic`** 选算法，**`workspace_`** 由 preference 上限约束。  
3. **缓存命中**：复用 **`matmul_desc` / A_desc / B_desc / D_desc / algo**。

### 3.2 每次调用

1. **`CUBLASLT_MATMUL_DESC_BIAS_POINTER`** 设为本次的 **`bias`**（每层指针可不同）。  
2. **`beta = 0`**：输出矩阵 **不累加** 原 `D` 内容（完全由 GEMM + bias 写入）。  
3. **`cublasLtMatmul`**：  
   - 标量 **`alpha`** 使用函数参数 **`alpha`**（SigLIP 为 **`a_qkv`**）。  
   - 矩阵实参顺序为 **`B, A_desc`** 与 **`A, B_desc`**（与 descriptor 的 **`(N,K)` × `(K,M) → (N,M)`** 约定对齐，从而在内存布局上等价于 **`[M,K] @ [K,N] → [M,N]`** 的语义；与仓库内 **`bf16_nn`** / pi05 FP8 路径同一套「列主 / 维度标签」习惯）。

---

## 4. Python 绑定

`flash_rt_kernels.GemmRunner.fp8_nn_bias(A, B, D, bias, M, N, K, alpha=1.0, stream=0)`  

参数为 **设备指针（uintptr）**；**`M,N,K` 与 C++ 一致**，**`alpha` 默认 1.0**，SigLIP 中显式传入 **`a_qkv`**。

声明与绑定：`csrc/gemm/gemm_runner.*`、`csrc/bindings.cpp`（`.def("fp8_nn_bias", ...)`）。

---

## 5. 计算流水线在层内的位置（SigLIP）

对每一层 `l`：

1. **`layer_norm_fp8`**：`x` → **`x_fp8`**（FP8 激活）。  
2. **`fp8_nn_bias`**：**`x_fp8 @ qkv_w + qkv_b` → `qkv`（FP16）**。  
3. **FMHA**：从 **`qkv`** 中按 stride 取 **Q/K/V**，写到 **`attn_out`**。  
4. 后续 **O 投影、FFN** 等使用 **`fp8_nn_bias_res` / `fp8_nn_gelu_bias`** 等带其它 epilogue 的接口。

---

## 6. 小结

| 项目 | 内容 |
|------|------|
| **作用** | **FP8×FP8 → FP16** 的矩阵乘，并在 cuBLASLt 内 **融合 bias**；标量 **`alpha`** 缩放乘积。 |
| **SigLIP 语义** | **`qkv = a_qkv * (x_fp8 @ W) + b`**，**`[S,3D]`** 输出。 |
| **后端** | **cuBLASLt** + **`EPILOGUE_BIAS`**；**描述符/算法按形状缓存**，**bias 指针按调用更新**。 |
| **与 `fp8_nn_dev`** | **`fp8_nn_bias`** 无 **A/B device scale**；**`alpha`** 承担调用侧传入的 **单层标量缩放**。 |

若需对照 **列主布局与 `B,A_desc` / `A,B_desc` 的严格等价证明**，可在文档中追加一页专门画 **`(N,K)×(K,M)`** 与 **`[S,D]@[D,3D]`** 的索引对应；当前工程内以 **`gemm_runner.cu` 注释**（`D = alpha * A(M,K) @ B(K,N) + bias(N)`）与 **`shared_primitives`** 注释为语义准绳。
