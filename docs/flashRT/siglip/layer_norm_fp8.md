# `layer_norm_fp8`：融合 LayerNorm + FP8 输出

本文说明 FlashRT 中 **`layer_norm_fp8` host 函数**及其在 FP16 路径上实际调用的 **`layer_norm_fp8_kernel_fp16`** 设备核：数学含义、数据流、启动配置，以及与 Python / SigLIP 流水线的关系。

实现位置：`third_party/FlashRT/csrc/kernels/norm.cu`（约 112–176 行）；声明：`csrc/kernels/norm.cuh`；Python 绑定：`csrc/bindings.cpp`（`m.def("layer_norm_fp8", ...)`）。

---

## 1. 作用（一句话）

对形状为 **`[seq_len, dim]`** 的 **FP16 输入**逐行做 **标准 LayerNorm（减均值、除标准差）**，再乘 **`gamma`**、加 **`beta`**（与 PyTorch `LayerNorm` 的 affine 一致），最后把结果 **cast 为 NV FP8 E4M3** 写入 **`out`**。  
即：**LayerNorm + affine 融合**，输出 dtype 为 **`__nv_fp8_e4m3`**，常用于 SigLIP 等路径里 **Norm 后直接喂 FP8 GEMM**，减少一次单独量化 kernel。

源码注释写明该 FP16 核与历史 **`pi05 fused_layernorm_fp8`** 行为对齐（verbatim production）。

---

## 2. 数学定义

对第 \(r\) 行（\(r = 0 \ldots R-1\)，\(R =\) `seq_len`），列维 \(C =\) `dim`，输入 \(x_{r,i}\)：

1. **均值**：\(\mu_r = \frac{1}{C}\sum_{i=0}^{C-1} x_{r,i}\)
2. **方差**：\(\sigma_r^2 = \frac{1}{C}\sum_i (x_{r,i} - \mu_r)^2\)
3. **标准化**：\(\hat{x}_{r,i} = (x_{r,i} - \mu_r) \cdot \frac{1}{\sqrt{\sigma_r^2 + \varepsilon}}\)
4. **Affine**：\(y_{r,i} = \hat{x}_{r,i} \cdot \gamma_i + \beta_i\)
5. **输出**：\(\texttt{out}_{r,i} = \mathrm{FP8\_E4M3}(y_{r,i})\)

其中 \(\gamma\)、\(\beta\) 为长度 \(C\) 的 **FP16** 向量（与 `gamma[i]`、`beta[i]` 逐元素对应）。

---

## 3. Host API 与启动方式

### 3.1 C 接口

```cpp
void layer_norm_fp8(const __half* x, __nv_fp8_e4m3* out,
                    const __half* gamma, const __half* beta,
                    int seq_len, int dim, float eps, cudaStream_t stream);
```

| 参数 | 含义 |
|------|------|
| `x` | 输入 FP16，逻辑形状 `[seq_len, dim]`，行主序 |
| `out` | 输出 FP8 E4M3，同形状连续存储 |
| `gamma` / `beta` | LayerNorm 的 weight / bias，长度 `dim`，FP16 |
| `seq_len` | 行数 \(R\) |
| `dim` | 列数 \(C\) |
| `eps` | **见下文 §6**：当前 FP16 专用核内部使用 **固定 `1e-6f`**，该参数在 **此 host 路径上未传入 device 核** |
| `stream` | CUDA stream |

Host 侧启动：

```text
layer_norm_fp8_kernel_fp16 <<< seq_len, 256, 0, stream >>>
    (x, out, gamma, beta, seq_len, dim);
```

- **Grid**：`seq_len` 个 block，**每个 block 负责一行**（`blockIdx.x == r`）。
- **Block**：256 线程；**动态共享内存为 0**（FP16 核用固定 `__shared__ float sh[32]` + warp shuffle 做归约）。
- **BF16 变体**：`layer_norm_fp8_bf16` 走模板核 `layer_norm_fp8_kernel<__nv_bfloat16>`，共享内存为 `256 * sizeof(float)`，且 **`eps` 会传入核**。

### 3.2 Python 绑定

`flash_rt_kernels.layer_norm_fp8(x, out, gamma, beta, seq_len, dim, eps=1e-6, stream=0)`  
参数为 **uintptr_t 设备指针**（与工程内其它 kernel 一致）。

---

## 4. `layer_norm_fp8_kernel_fp16` 内部流程（按执行顺序）

核函数签名中 `R`、`C` 即 `seq_len`、`dim`。

1. **行边界**：`r = blockIdx.x`，若 `r >= R` 则返回。
2. **指针定位**：`row = in + r * C`，`orow = out + r * C`，本 block 只处理第 `r` 行。
3. **求和（算均值）**  
   - 每个线程对列下标 `i = threadIdx.x, threadIdx.x + blockDim.x, ...` 累加 `__half2float(row[i])`。  
   - 通过 **warp 内 `__shfl_xor_sync`** 与 **`sh[32]` + `__syncthreads`**** 的两级归约，把整行和收到 **thread 0 的 `sh[0]`**。  
   - `mean = sh[0] / C`。
4. **求方差**  
   - 各线程累加 `(row[i] - mean)^2`，同样 shuffle + shared 归约到 `sh[0]`。  
   - `rstd = rsqrtf(sh[0] / C + 1e-6f)` — **此处 \(\varepsilon\) 写死为 `1e-6f`**。
5. **仿射 + 写 FP8**  
   - 对每个 `i`：`normed = ((x-mean)*rstd) * gamma + beta`（在 float 域算）。  
   - `orow[i] = __nv_fp8_e4m3(normed)` — 硬件/编译器支持的 FP8 转换。

**要点**：一行内的 **mean / var** 对该行所有线程可见后，再写输出；行与行之间 **无数据依赖**，完全由 **grid 维** 并行。

---

## 5. 与「先 `layer_norm_fp16` 再量化」的差异

| 方式 | 特点 |
|------|------|
| `layer_norm_fp8` | 单 kernel：**LN + affine + FP8 cast**；中间归一化结果可不落盘 FP16 全宽（仍用 float 累加归约）。 |
| `layer_norm_fp16` + 单独 `quantize_*` | 两阶段；多一次全局内存读写与调度。 |

在 **SigLIP Thor** 路径中，`shared_primitives.siglip_forward` 对 attention / FFN 前的 Norm 调用 **`fvk.layer_norm_fp8(...)`**，输出写入 **`x_fp8` / 等 FP8 缓冲**，紧接着 **`gemm.fp8_nn_*`**，与该融合设计一致。

---

## 6. 关于 `eps` 参数（实现细节）

- **`layer_norm_fp8`（FP16 → FP8）** 实际调用的 **`layer_norm_fp8_kernel_fp16` 在方差归一化里固定使用 `1e-6f`**，与 host 形参 **`eps` 解耦**。  
- 若调用方传入非 `1e-6` 的 `eps`，**当前 FP16 核行为不会改变**。  
- 需要 **可配置 `eps`** 且 BF16 输入时，应使用 **`layer_norm_fp8_bf16`** 路径（模板核使用 `block_reduce_sum` 与传入的 `eps`）。

---

## 7. 相关代码索引

| 位置 | 内容 |
|------|------|
| `norm.cu` 112–138 | `layer_norm_fp8_kernel_fp16` |
| `norm.cu` 171–176 | `layer_norm_fp8` host launch |
| `norm.cu` 140–181 | BF16 模板核 + `layer_norm_fp8_bf16` |
| `flash_rt/hardware/thor/shared_primitives.py` | `siglip_forward` 内 `fvk.layer_norm_fp8(x, x_fp8, ln_*, S, D, 1e-6, stream)` |

---

## 8. 小结

- **`layer_norm_fp8`**：SigLIP / 类似流水线中 **Norm 与 FP8 激活衔接** 的融合入口（FP16 权重与输入）。  
- **流程**：按行求 mean → var → `inv_std` → affine → **E4M3 输出**。  
- **并行**：**一行一个 CUDA block**，列方向上线程协作归约，再并行写 FP8。  
- **注意**：FP16 核 **eps 固定为 1e-6**；与 Python 绑定默认值一致，但与 host `eps` 形参在字面上不完全等价。
