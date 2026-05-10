# SigLIP 中的 `fmha_strided_full`：交错 QKV 上的多视角 FMHA

本文对应 `shared_primitives.siglip_forward` 在 **`attn is None`** 分支里对 **`fvk.fmha_strided_full(...)`** 的调用（约 114–120 行），说明 **内存布局**、**Q/K/V 指针如何从 `qkv` 推出**、**各参数含义**，以及与 **`ThorFlashAttnBackend`** 路径的关系。

---

## 1. 在层内处于什么位置

单层 attention 内顺序大致为：

1. **`layer_norm_fp8`** → 激活写入 **`x_fp8`**。  
2. **`gemm.fp8_nn_bias`** → **`qkv`**（FP16），逻辑形状 **`[S, 3·D]`**，其中 **`D = NH · HD`**（SigLIP 上为 1152 = 16×72）。  
3. **本 kernel**：从交错 **`qkv`** 读 Q/K/V，写 **`attn_out`**（FP16，`[S, D]`）。  
4. **`quantize_fp8_static_fp16`** 等后续步骤。

`siglip_forward` 中 **`scale = 1.0 / sqrt(HD)`** 仅被计算，**未传入** Python 的 **`fmha_strided_full`**。底层 **`fmha_fp16_strided`** 的 **`Arguments`** 里传入一组与 **`fmha_fp16_attn`** 相同的标量（见 `fmha_fp16_strided.cu` 111–112 行）；**Softmax 前的 \(1/\sqrt{d_k}\)** 由 CUTLASS FMHA **collective / kernel 模板**侧处理，**不**由 `shared_primitives` 的 `scale` 变量控制。若删除 Python 里该行，不改变当前 FMHA 数值。

---

## 2. `qkv` 的内存布局（交错 QKV）

GEMM 输出约定为 **行主序 FP16**，**一行一个 token**，列维 **`3·D`**：

| 列区间（逻辑） | 内容 | 长度 |
|----------------|------|------|
| `[0 : D)` | **Q** | `D` |
| `[D : 2D)` | **K** | `D` |
| `[2D : 3D)` | **V** | `D` |

记 **`stride = 3·D`**（单位：**元素个数**，与代码中 `stride = 3 * D` 一致）。  
相邻两个 token（沿 **`S`** 维）在内存中相隔 **`stride` 个 FP16**，即 **`stride · sizeof(fp16) = stride · 2` 字节**。

总 token 数 **`S = num_views · seq_per_view`**（例如 2×256）：**视角 0** 的 `seq_per_view` 行接着 **视角 1** 的 `seq_per_view` 行，中间无额外填充（由 buffer 分配保证）。

**`D` 与多头**：**`D = NH · HD`**。每个 Q/K/V 块内部再按 head 切 **`HD`** 维；strided FMHA 接口用 **`nheads_q` / `nheads_kv` / `head_dim`** 描述 head 划分，与连续 **`D`** 维一致。

---

## 3. Q / K / V 指针为何是「基址 + 字节偏移」

Python 里传入的是 **整数设备地址**（`data_ptr()`）。**FP16 元素占 2 字节**。

- **`Q_ptr = qkv`**  
  指向缓冲区的 **首字节**，即 **token 0 的 Q 段第一个元素**。

- **`K_ptr = qkv + D * 2`**  
  从首地址前进 **`D` 个 FP16** → **`D·2` 字节**，落到 **同一 token 行内 K 段** 的起点（紧接在 Q 块之后）。

- **`V_ptr = qkv + 2 * D * 2`**  
  再前进 **`D` 个 FP16**，落到 **同一行 V 段** 的起点。

因此：**三个指针都锚在「token 0」这一行的三个块上**；沿序列维走到 token `t` 的 Q/K/V，由 **strided FMHA** 用 **`stride_q` / `stride_kv`** 从各自基址步进，而不是再为整段 `qkv` 另算三个独立 tensor。

这与 **`ThorFlashAttnBackend`** 中 siglip 分支的算法一致（`attn_backend.py` 221–227 行）：`K = Q + D*2`，`V = Q + 2*D*2`，`stride = 3*D`，再调同一个 **`fmha_strided_full`**。

---

## 4. `fmha_strided_full` 调用逐项（114–120 行）

```python
stride = 3 * D  # QKV interleaved stride
Q_ptr = qkv
K_ptr = qkv + D * 2    # byte offset for fp16
V_ptr = qkv + 2 * D * 2
fvk.fmha_strided_full(Q_ptr, K_ptr, V_ptr, attn_out,
                      nv, spv, spv, NH, NH, HD,
                      stride, stride, stream)
```

### 4.1 指针参数

| 参数 | 含义 |
|------|------|
| **`Q` / `K` / `V`** | 上文三个基址；**元素语义**为 FP16，**布局**为「基址 + 大 stride 的 2D 栅格」。 |
| **`O`** | **`attn_out`**，FMHA 输出 **FP16**，逻辑 **`[S, D]`**，与后续 O 投影输入一致。 |

### 4.2 形状 / 并行语义

| 参数 | SigLIP 典型取值 | 含义 |
|------|-----------------|------|
| **`batch`** | **`nv`**（`num_views`） | **batch 维 = 视角数**：每个视角一段 **`seq_per_view`** token，**各自独立做 self-attention**（不跨视角混注意力）。 |
| **`seq_q`** | **`spv`**（如 256） | 每个 batch 条目的 **query 序列长度**。 |
| **`seq_kv`** | **`spv`** | 与 **`seq_q`** 相同 → **自注意力**，KV 序列长与 Q 一致。 |
| **`nheads_q`** | **`NH`** | Query 头数。 |
| **`nheads_kv`** | **`NH`** | SigLIP 为 **MHA**，K/V 头数与 Q 相同（非 GQA）。 |
| **`head_dim`** | **`HD`**（如 72） | 每头维数。 |

### 4.3 Stride

| 参数 | 取值 | 含义 |
|------|------|------|
| **`stride_q`** | **`3*D`** | 沿序列前进 **一个 token** 时，在 **Q 平面**上指针应前进的 **元素**步长（此处等于一整行 QKV 宽度，因 Q 块后接 K、V，下一 token 的 Q 正好隔 **`3·D`** 个 FP16）。 |
| **`stride_kv`** | **`3*D`** | K、V 相对各自基址沿序列的步长；本布局下与 **`stride_q`** 相同。 |

**直觉**：把 **`qkv`** 看成 **`[S, 3D]`** 的大矩阵，但 Q/K/V kernel 各从一个 **子平面起点** 读数，用 **`stride=3D`** 在「下一 token」对齐到各自的 Q/K/V 列块。

---

## 5. C / Python 侧接口与实现落点

- **声明**：`csrc/attention/fmha_dispatch.h`（`fmha_strided_full`）。  
- **分发**：`csrc/attention/fmha_dispatch.cu` — 若已通过 **`load_fmha_strided_library`** 解析到符号 **`fmha_fp16_strided`**，则把指针与整数参数原样转发（与动态库导出签名一致）。  
- **具体实现（源码）**：**`csrc/attention/fmha_fp16_strided.cu`** 中 **`extern "C" int fmha_fp16_strided`**（约 85–125 行）。工程既可把该 TU **链进 `libfmha_fp16_strided.so`** 供 `dlopen`，也可与主扩展一并链接；逻辑以该文件为准。  
- **Python**：`flash_rt_kernels.fmha_strided_full(...)`（`bindings.cpp` 1135–1145 行附近）。

未加载 strided 库、或 **`can_implement` / `initialize` / `run` 失败** 时，分发层或该 `.cu` 会返回负错误码（见 `fmha_dispatch.cu` 与 `fmha_fp16_strided.cu` 中 `-1` / `-2` / `-3`）。

---

## 6. 源码实现：`fmha_fp16_strided`（`fmha_fp16_strided.cu`）

本节对应 **`fmha_fp16_strided.cu`** 中与 **`fmha_fp16_attn`**（连续 QKV）并列的 **Strided API**（80–125 行）。

### 6.1 后端与算子类型

- 使用 **CUTLASS FMHA** 设备算子：**`cutlass::fmha::device::FMHA<Kernel>`**，其中 **`Kernel`** 为 **`Sm100FmhaFwdKernelTmaWarpspecialized`**（TMA、warp-specialized mainloop / epilogue / load），**`TileShape = Shape<_256,_128,_128>`**。  
- 元素类型：**`Element = cutlass::half_t`**（FP16），累加类型 **`ElementAccQK` / `ElementAccPV` = float**，输出 **`ElementOut = half_t`**。  
- 头文件依赖 **`sm100_fmha_*`** 与 **`cute::tensor`**：面向 **SM100 族**（Thor **SM110** 产品路径与构建脚本需与之匹配；若 GPU 架构不符，`can_implement` 会失败）。

### 6.2 问题规模 `ProblemShape` / 头数拆分

```cpp
int H_Q = NQ / NKV, H_K = NKV, H = H_Q * H_K;
int D = cutlass::round_up(HD, 8);
auto ps = cute::make_tuple(SQ, SK, D, cute::make_tuple(cute::make_tuple(H_Q, H_K), B));
```

| 符号 | SigLIP（`NQ = NKV = NH`） | 含义 |
|------|---------------------------|------|
| **`H_Q`** | **`NH / NH = 1`** | GQA 语义下的 **Q 组数**；MHA 时为 1。 |
| **`H_K`** | **`NH`** | **KV 头数**（此处等于 Q 头数）。 |
| **`H`** | **`NH`** | 逻辑总头数 **`H_Q·H_K`**。 |
| **`D`** | **`round_up(HD, 8)`** | Kernel 内 **K 维对齐到 8**（与 Tile / TMA 约束一致），**`HD` 通常已为 8 的倍数（如 72）**。 |
| **`ps`** | **`(SQ, SK, D, ((H_Q,H_K), B))`** | 序列长、头分组、**batch=`B`**（SigLIP 的 **`nv`**）。 |

### 6.3 Stride：交错读 Q/K/V vs 连续写 O

**Q 的 stride `sQ`**（与 token 间距 **`q_seq_stride`** 绑定）：

```cpp
StrideQ sQ = make_stride(q_seq_stride, _1{},
    make_stride(make_stride(D, H_Q * D), q_seq_stride * SQ));
```

- 最外层「沿序列」步长为 **`q_seq_stride`（元素）**：SigLIP 传 **`3*D`**，即 **从某一 token 的 Q 块首地址，跳到下一 token 的 Q 块首地址** 跨过整行 **`[Q‖K‖V]`**。  
- 内层 `make_stride(D, H_Q*D)` 等描述 **头维 / 特征维** 在 **单个 Q 块** 内的布局（与 **`H_Q=1`** 的 MHA 一致）。

**K / V 的 stride `sK`**（与 **`k_seq_stride`** 绑定；**K 与 V 共用同一 `StrideK`**，因交错布局下 K、V 相对各自基指针的步进规律相同）：

```cpp
StrideK sK = make_stride(k_seq_stride, _1{},
    make_stride(make_stride(_0{}, D), k_seq_stride * SK));
```

- **`k_seq_stride`**：SigLIP 与 Q 相同，为 **`3*D`**。  
- **`_0{}`** 处与连续版 **`fmha_fp16_attn`** 的 K stride 不同：连续 K 在 head 维上有 **`H_K*D`** 一类步长；**交错布局** 下 K/V 子张量通过 **不同的基指针 `K`/`V`（相差 `2·D` 个 FP16）** 对齐到各自列块，故 stride 结构用 **`_0`** 占位配合 **`k_seq_stride`**。

**输出 O 的 stride `sO`**（**始终为连续 `[S, NH, HD]`**）：

```cpp
StrideO sO = make_stride(H * D, _1{},
    make_stride(make_stride(D, H_Q * D), H * D * SQ));
```

即 FMHA 将 **`attn_out`** 写成 **标准密集布局**，供后续 **O 投影** 等消费；**不再**保持 `3*D` 交错。

### 6.4 LSE、workspace 与设备属性

- **`SQ_r = ((SQ + 127) / 128) * 128`**：序列维 **向上取整到 128**，用于 **LSE（log-sum-exp）** 缓冲布局 **`StrideLSE sL`**。  
- **`g_lse`**：全局 **`cudaMalloc`** 缓存，大小 **`B * H * SQ_r * sizeof(float)`**，不足则 **释放重配**。  
- **`g_ws`**：**`FmhaOp::get_workspace_size`** 得到的 **workspace**，同样按需扩容。  
- **`cudaDeviceGetAttribute(..., cudaDevAttrMultiProcessorCount, 0)`**：把 **SM 数量** 传入 **`Arguments`** 的调度子结构 **`{0, sm}`**（与 `fmha_fp16_attn` 相同）。

### 6.5 启动与错误码

1. **`op.can_implement(args)`**：不满足 Tile/对齐/架构等时返回 **`kError`**，strided 路径会 **`printf`** 打印 **`qstride`/`kstride`** 并返回 **`-1`**。  
2. **`op.initialize(args, g_ws, stream)`**：失败返回 **`-2`**。  
3. **`op.run(stream)`**：失败返回 **`-3`**。  
4. 成功返回 **`0`**。

### 6.6 与 `fmha_fp16_attn`（同文件 48–78 行）的对比

| 项目 | **`fmha_fp16_attn`** | **`fmha_fp16_strided`** |
|------|----------------------|-------------------------|
| **Q/K/V 布局** | 各自 **连续** `[S, …]` | **交错**：通过 **基指针偏移 + `q_seq_stride` / `k_seq_stride`** |
| **`StrideQ` 首元** | **`H*D`**（dense） | **`q_seq_stride`**（如 **`3*D`**） |
| **`StrideK` 首元** | **`H_K*D`** | **`k_seq_stride`** |
| **额外参数** | 无 stride 形参 | **`q_seq_stride`, k_seq_stride`** |
| **其余** | 同一 **`FmhaOp`**、`ps` 构造方式、`g_lse`/`g_ws`、标量组 | 相同 |

---

## 7. 与 `if attn is not None` 分支的关系

- **`attn is None`**：本文件 **直接** 计算 **`Q_ptr`/`K_ptr`/`V_ptr`** 并调用 **`fmha_strided_full`**（本文档描述的路径）。  
- **`attn is not None`**：**`attn.run("siglip", ...)`** 内部执行 **同一套** 指针与 stride 逻辑，再调 **`fmha_strided_full`**（便于与 encoder/decoder 统一 **AttentionBackend** 协议、槽位校验与测试）。

数值上两条路径应对齐；差异主要在 **是否** 经过 backend 封装。

---

## 8. 小结表

| 概念 | 说明 |
|------|------|
| **`qkv` 布局** | **`[S, 3D]`** FP16，行主序；每行 **`[Q‖K‖]`** 各 **`D`** 维。 |
| **`Q_ptr`** | **`qkv` 起点**（token0 的 Q）。 |
| **`K_ptr`** | **`qkv + 2·D` 字节**（跳过 **`D` 个 FP16** 的 Q）。 |
| **`V_ptr`** | **`qkv + 4·D` 字节**（再跳过 **`D` 个 FP16** 的 K）。 |
| **`stride`** | **`3·D`（元素）**：token 维上 Q/K/V 子平面步进一致。 |
| **`batch = nv`** | 多视角 **独立** FMHA，每视角 **`spv`** token。 |
| **后端** | **`fmha_fp16_strided.cu`**：`fmha_fp16_strided` → CUTLASS **`FmhaOp::run`**；经 **`fmha_dispatch::fmha_strided_full`** / Python **`fmha_strided_full`** 调用。 |

---

## 9. `op` 如何「指定」Softmax？——精确到仓库内源码

### 9.1 为何在 `fmha_fp16_strided.cu` 里看不到 `softmax`

**`fmha_fp16_strided`**（85 行起）只做三件事：**拼 `Arguments` → `initialize` → `run`**。  
**Softmax 不是可选 epilogue**，而是 **`cutlass::fmha::device::FMHA<Kernel>`** 所实例化的 **`Kernel` / `Mainloop`** 里 **固定的数值流水线**；因此 **不会** 在封装 `.cu` 里出现 `softmax(...)` 字样的调用。

```110:124:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/attention/fmha_fp16_strided.cu
    typename FmhaOp::Arguments args{ps,
        {{(Element const*)Q, sQ, (Element const*)K, sK, (Element const*)V, sK},
         0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {(ElementOut*)O, sO, g_lse, sL}, {0, sm}};

    FmhaOp op;
    ...
    if (op.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -2;
    return (op.run(stream) == cutlass::Status::kSuccess) ? 0 : -3;
```

**`op.run(stream)`** 进入 **`Sm100FmhaFwdKernelTmaWarpspecialized`** 的设备代码，其中 **mainloop 阶段**会调用设备函数 **`softmax` / `softmax_step`**（见下一小节）。

---

### 9.2 第一层：用 **C++ 类型** 选定「带 Softmax 的 FMHA」算子族

在 **`fmha_fp16_strided.cu`** 顶部，**`Mainloop`** 别名已经把算法族写死为 **「前向 FMHA + TMA + warp-specialized」**，其中 **Softmax 是 mainloop 内建步骤**：

```34:42:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/attention/fmha_fp16_strided.cu
using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementAccQK, ElementAccPV, TileShape,
    StrideQ, StrideK, StrideV, cutlass::fmha::collective::NoMask>;
...
using FmhaOp = cutlass::fmha::device::FMHA<Kernel>;
```

- **`Sm100FmhaFwdMainloopTmaWarpspecialized<..., NoMask>`**：这里的 **`NoMask`** 只影响 **是否需要额外 masked softmax 迭代**（见 9.4），**不是**「关闭 softmax」。  
- 若把 **`Mainloop`** 换成别的 collective，才可能换成别的注意力变体；**当前 FlashRT SigLIP 路径没有这种运行时开关**。

---

### 9.3 第二层：用 **`Mainloop::Arguments` 里的标量** 控制 **缩放系数**（含默认 \(1/\sqrt{d}\)）

FlashRT 仓库内 **`Mainloop` 的 `Arguments` 定义**在（注意：该头在 **`third_party/FlashRT/csrc/attention/collective/`** 下，与 `fmha_fp16_strided.cu` 的 `#include "collective/..."` 一致）：

```192:205:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/attention/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };
```

**`fmha_fp16_strided`** 与 **`fmha_fp16_attn`** 传入的第二组均为 **`{0.0f, 1.0f, 1.0f, 1.0f, 1.0f}`**（见 `fmha_fp16_strided.cu` 68–69、111–112 行），对应 **`Mainloop::Arguments`** 中 **`scale_softmax` + `scale_{q,k,v}` + `inv_scale_o`** 共 **5 个标量**；语义上关键是：

- **`scale_softmax = 0.0f`**：表示 **「使用默认 attention 缩放」**。  
- 在 **`to_underlying_arguments`** 里 **把 0 解析成 \(1/\sqrt{\texttt{get<2>(problem\_shape)}}\)**（`problem_shape` 即 **`ps`** 的第三维 **`D = round_up(HD,8)`**）：

```227:244:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/attention/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / (float) std::sqrt(get<2>(problem_shape));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));
    ...
    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o / kPRescale
    };
```

因此：**「要做 softmax」由算子族保证；「\(QK^\top\) 前的缩放」由 `scale_softmax`（0→默认 \(1/\sqrt{D}\)）与 `scale_q*scale_k` 在 `Params` 里合成到 `scale_softmax` / `scale_softmax_log2`。** SigLIP 路径 **`scale_q=scale_k=1`**，等价于 **标准 \(1/\sqrt{d_k}\)**（在 **`HD` 为 8 的倍数** 时 **`D` 与 `HD` 一致**）。

---

### 9.4 第三层：设备上 **Softmax 本体** 在 `softmax_step` / `softmax` 里

同一文件中的 **`softmax_step`** 从 TMEM 读出 logits **`S`**（即 **`QK^\top`** 的 tile），做 **行方向 `row_max`**，再用 **`scale_softmax_log2`** 做 **`fma` + `exp2f`** ——这是 **以 2 为底的指数**，等价于在 log2 域实现 **\(\exp(\alpha(S - m))\)** 的 softmax 分子部分；随后 **`row_sum`** 累加完成 **归一化所需的分母信息**（**在线 softmax / tile 迭代** 风格）：

```586:647:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/attention/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
    ElementQK old_row_max = row_max;
    {
      // compute rowmax
      float row_max_0 = row_max;
      ...
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        ...
      row_max = ::fmax(row_max_0, row_max_1);
      ...
    }
    ...
    ElementQK scale = params.scale_softmax_log2;
    ElementQK row_max_scale = row_max_safe * scale;
    ...
    for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
      ...
      cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
      ...
      tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
      tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));
      ...
    }
```

外层的 **`softmax(...)`**（约 730 行起）根据 **`Mask`** 的 **`get_unmasked_trip_count` / `get_masked_trip_count`** 循环调用 **`softmax_step<false>`** 与 **`softmax_step<true>`**；**`NoMask`** 时 masked 分支的 trip count 通常为 0，主要走 **无 mask** 路径。

---

### 9.5 和 PyTorch `eager_attention_forward` 的对照（复习）

- **PyTorch**：`matmul(Q,K^T)*scaling` → `softmax` → `matmul(·,V)`，显式三步。  
- **本 FMHA**：**同一数学语义**被 **融合进 `FmhaOp::run` → mainloop 的 `softmax`/`softmax_step` + 后续 PV**，**缩放**由 **`scale_softmax`（默认 \(1/\sqrt{D}\)）** 与 **`scale_q*scale_k`** 注入 **`Params`**，**不是** `shared_primitives.py` 里未使用的 **`scale = 1/sqrt(HD)`** 变量。

---

### 9.6 一句话收束

| 问题 | 答案（精确到代码层级） |
|------|------------------------|
| **谁在「指定」要做 softmax？** | **`Kernel`/`Mainloop` 模板名**：`Sm100FmhaFwdMainloopTmaWarpspecialized` **内含** softmax 流水线；**不是** `fmha_fp16_strided` 里 if 开关。 |
| **缩放 \(1/\sqrt{d}\) 从哪来？** | **`Mainloop::Arguments::scale_softmax == 0`** → **`to_underlying_arguments`** 设为 **`1/sqrt(get<2>(problem_shape))`**，再乘 **`scale_q*scale_k`** 写入 **`Params`**。 |
| **softmax 实现在哪几行？** | **`sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`** 的 **`softmax_step` / `softmax`**（上引片段及 730 行附近）。 |
| **`op.run` 控制什么？** | **整块 FMHA 的调度与融合执行**；**计算过程**由 **`ProblemShape` + stride + `Params` + Tile/Scheduler** 共同决定，**softmax 是其中固定子阶段**。 |
