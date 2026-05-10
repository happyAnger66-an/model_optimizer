# `patch_embed_bias_pos`：Bias + 位置编码融合核

源码：`third_party/FlashRT/csrc/kernels/patch_embed.cu`（设备核 + host 封装）、声明 `patch_embed.cuh`、PyBind `csrc/bindings.cpp`（`fvk.patch_embed_bias_pos`）。

## 1. 在整条链路中的位置

SigLIP / Paligemma 视觉入口典型顺序为：

1. **`patch_im2col`**：图像 `(nv, 224, 224, 3)` → patches `(nv×256, 588)` FP16。  
2. **FP16 GEMM**：patches `@` patch embedding 权重 → **`output` 形状 `(S, D)`**，其中 **`S = nv × 256`**，`D` 为 SigLIP hidden dim（如 1152）。  
3. **`patch_embed_bias_pos`**：在 **已有 GEMM 输出** 上 **原地** 加上 **patch bias** 与 **2D 位置编码**（按 token 在单视图内的下标取 `pos_emb`）。

文件头注释给出的数学语义：

```4:5:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/patch_embed.cu
// 2. bias_pos: output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
```

即：**同一视图内 256 个 patch 共享一套位置向量**；**多视图**时，第 2 个视图的 token 下标 `i` 与第 1 个视图在 **`i % S_per_view`** 上对齐到同一行位置表（**视图间位置编码复用**）。

---

## 2. Host 封装：`patch_embed_bias_pos`

```79:87:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/patch_embed.cu
void patch_embed_bias_pos(half* output, const half* bias, const half* pos_emb,
                          int S, int D, int S_per_view, cudaStream_t stream)
{
    int total = S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_embed_bias_pos_kernel<<<blocks, threads, 0, stream>>>(
        output, bias, pos_emb, S, D, S_per_view);
}
```

| 参数 | 含义 |
|------|------|
| `output` | **in-out**。形状逻辑 **`[S, D]`**，行主序展平为 **`S*D` 个 `half`**；进入核时已是 GEMM 结果，核内在其上做 **`+=`**。 |
| `bias` | **`[D]`** 的 FP16 patch embedding bias。 |
| `pos_emb` | **`[S_per_view, D]`** 展平为 **`S_per_view * D` 个 `half`**，行主序：`pos_emb[pos_i * D + j]` 即第 `pos_i` 个 patch、第 `j` 维。 |
| `S` | 序列长度（所有视图 patch 拼在一起），一般为 **`nv * 256`**。 |
| `D` | 隐藏维度。 |
| `S_per_view` | **单视图 patch 数**，SigLIP 固定为 **256**（16×16 grid）。 |
| `stream` | CUDA stream，与前后 `patch_im2col` / GEMM 同流以保证顺序。 |

**启动配置**：**一维 grid**，总线程数覆盖 **`S * D`** 个元素；每块 **256** 线程，`blocks = ceil((S*D)/256)`。无共享内存。

---

## 3. 设备核：`patch_embed_bias_pos_kernel`

```59:77:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/patch_embed.cu
__global__ void patch_embed_bias_pos_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ pos_emb,
    int S, int D, int S_per_view)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * D;
    if (idx >= total) return;

    int i = idx / D;
    int j = idx % D;
    int pos_i = i % S_per_view;

    float v = __half2float(output[idx])
            + __half2float(bias[j])
            + __half2float(pos_emb[pos_i * D + j]);
    output[idx] = __float2half(v);
}
```

### 3.1 线程到 `(i, j)` 的映射

- **`idx`**：全局线性下标，`0 … S*D - 1`。  
- **`i = idx / D`**：token / patch 在 **全长序列** 中的行号 **`[0, S)`**。  
- **`j = idx % D`**：特征维 **`[0, D)`**。  
- **`pos_i = i % S_per_view`**：把全局行号 **折叠到单视图内的 patch 索引** **`[0, S_per_view)`**。

因此 **第 `i` 行**使用的位置向量与 **第 `pos_i` 行**的 `pos_emb` 一致；**不同视图**上相同网格位置的 patch **共用** `pos_emb` 的同一行（与 HF Paligemma / SigLIP 常用设定一致：**位置表长度 256，按 patch 在图像网格中的位置索引**）。

### 3.2 数值计算

对每个 **`(i, j)`**：

\[
\text{output}[i,j] \leftarrow \text{fp16}\Big(
  \text{fp32}(\text{output}[i,j]) + \text{fp32}(\text{bias}[j]) + \text{fp32}(\text{pos\_emb}[\text{pos\_i}, j])
\Big)
\]

实现上：**三者先转为 `float` 相加，再 `__float2half` 写回**。  
作用：**在 FP32 中完成加法**，减轻 **FP16 累加顺序/舍入** 与纯半精度链式加法的差异；仍 **最终以 FP16 存 output**。

### 3.3 内存访问模式

- **`output[idx]`**：合并访问取决于 **`idx` 连续时 `i` 固定、`j` 递增** → 行内 **`j` 连续** 时相邻线程访问 **`output` 连续**，合并度好。  
- **`bias[j]`**：同一线程块内 **`j`** 可能不同 → **广播式随机**访问 bias（维度 D 通常较大，cache 友好程度一般）。  
- **`pos_emb[pos_i * D + j]`**：`pos_i` 由 `i` 决定，块内不同线程可能落在 **不同 `pos_i`** → 访问 **`pos_emb` 的不同行**；同一 warp 内若 `i` 相近则 **`pos_i` 可能相同**，有一定重用。

---

## 4. Python 绑定

```662:669:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/bindings.cpp
    m.def("patch_embed_bias_pos", [](uintptr_t output, uintptr_t bias, uintptr_t pos_emb,
                                      int S, int D, int S_per_view, uintptr_t stream) {
        patch_embed_bias_pos(reinterpret_cast<half*>(output),
                             reinterpret_cast<const half*>(bias),
                             reinterpret_cast<const half*>(pos_emb),
                             S, D, S_per_view, to_stream(stream));
    }, py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
       py::arg("S"), py::arg("D"), py::arg("S_per_view"), py::arg("stream") = 0);
```

前端（如 `pi05_thor._patch_embed_ops`）在 **`fp16_nn(patches, pe_w, sig_x, ...)`** 之后调用 **`fvk.patch_embed_bias_pos(sig_x, pe_b, pos_emb, S_sig, D_sig, 256, stream)`**，与 `D_sig`、`S_sig = nv*256` 及 **`pos_emb` 只存 256 行** 的加载方式一致。

---

## 5. 小结

- **`patch_embed_bias_pos`** 对 **patch 线性层后的 FP16 输出** 做 **原地融合**：**加 patch bias + 加（按 `i % 256` 索引的）位置编码**。  
- **多视图**时通过 **`i % S_per_view`** 实现 **256 行位置表在视图间复用**。  
- 核内 **FP32 累加再写回 FP16**，与 **逐元素并行、无共享内存** 的简单 launch 策略，便于与 **im2col + GEMM** 拼进 **CUDA Graph**。
