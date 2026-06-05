# C1：`fused_adarms_fp8_static_fp16` 算子说明

> **实现**：`third_party/FlashRT/csrc/kernels/decoder_fused.cu::fused_adarms_fp8_static_fp16_kernel`  
> **Host 入口**：`fused_adarms_fp8_static_fp16`（`<<<S, 256>>>`）  
> **Python 绑定**：`fvk.fused_adarms_fp8_static_fp16`（`flashrt_decoder/kernels.py` 加载）  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` 每层 C1  
> **融合设计背景**：[`docs/optimizer/flashrt/fusion_design.md`](../../optimizer/flashrt/fusion_design.md) §4

---

## 1. 功能概述

FlashRT **C1 融合 kernel** 将 OpenPI/Pi0.5 decoder 中 self-attention **之前** 的三步合并为一次 GPU 调用：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | **AdaRMSNorm** | `y = RMSNorm(x) * (1 + scale) + shift` |
| 2 | **静态 FP8 量化** | `y_fp8 = clamp(y / act_scale, [-448, 448])` |
| 3 | **Gate 旁路输出** | 将 `gate` 写入 buffer，供后续 C4→C5 的 gated residual 使用 |

PyTorch/OpenPI 中对应逻辑（Expert 分支）：

```python
# gemma_pytorch.py — 仅 expert 分支
hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

# modeling_gemma.py — GemmaRMSNorm
modulation = self.dense(cond)           # Linear(Da → 3*D)
scale, shift, gate = chunk(modulation, 3)
x_norm = rmsnorm(x) * (1 + scale) + shift
# 随后 q_proj / k_proj / v_proj(x_norm)
```

FlashRT 一次 kernel 完成 norm + 量化 + gate 落盘，norm 结果以 **FP8** 形式直接进入 C2 QKV GEMM，避免 `[S, D]` fp16 中间激活写回 DRAM。

---

## 2. 张量布局

| 参数 | 形状 |  dtype | 含义 |
|------|------|--------|------|
| `x` | `[S, D]` | fp16 | 进 self-attn 前的残差流 |
| `style`（`sa_ptr`） | `[S, 3*D]` | fp16 | 预计算的 Ada 调制：`[scale \| shift \| gate]` 按行拼接 |
| `out`（`xn_fp8`） | `[S, D]` | fp8 (E4M3) | 量化后的 norm 输出，供 C2 QKV FP8 GEMM |
| `gate_out`（`gate`） | `[S, D]` | fp16 | gate 向量，持久化到 `bufs['gate']` |
| `descale_ptr` | 标量（device） | fp32 | 本层 QKV 路径静态 `act_scale`（标定 amax），槽位 `l*4+0` |

- `S`：action token 数（Pi0.5 扩散步中通常为 10）
- `D`：隐藏维（≈1024）
- `style` 每行布局：`sc = style[r*3*D : r*3*D+D]`，`sh = sc+D`，`gt = sh+D`

---

## 3. 数学公式

### 3.1 AdaRMSNorm（Gemma 风格，无均值中心化）

与 LayerNorm 不同，RMSNorm **不减均值**，只做 RMS 缩放：

```
rms  = sqrt(mean(x²) + ε),   ε = 1e-6
y[i] = x[i] / rms * (1 + scale[i]) + shift[i]
```

实现中用 `rstd = rsqrt(mean(x²) + ε)` 代替 `1/rms`，等价于 `x[i] * rstd`。

### 3.2 静态 FP8 量化（E4M3）

```
inv_scale = 1 / max(act_scale, 1e-12)
y_fp8[i]  = clamp(y[i] * inv_scale, -448, 448)   →  __nv_fp8_e4m3
```

- **E4M3**：4 位指数 + 3 位尾数，最大有限值 ≈ **448**
- **静态 per-tensor scaling**：`act_scale` 在导出/标定阶段固定，**非**运行时动态 amax
- 与后续 cuBLASLt FP8 GEMM 的 `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` 配套；C2 descale 使用同一 `act_scale_qkv`

### 3.3 Gate 输出

Gate 不参与 C1 内的计算，仅 **拷贝** 到 `gate_out`：

```
gate_out[i] = gate[i]   （style 的第三段）
```

后续 C4→C5 `gate_res_adarms_fp8_static_fp16` 语义：`x ← x + o_proj_out * gate`。

---

## 4. 计算三阶段

Kernel 对每个 row（`blockIdx.x = r`）分三阶段执行：

```text
Phase 1: 各线程 stride 累加 sum(x²)
    ↓
Phase 2: warp/block 两级归约 → sum_sq → rstd
    ↓
Phase 3: 归一化 + Ada 仿射 + FP8 量化 + 写 gate
```

| 阶段 | 操作 | 关键技术 |
|------|------|----------|
| **Phase 1** | 各线程按 `i, i+256, i+512, …` 累加 `sum(x²)` | `__half2float` 在 FP32 中计算，避免大 D 时 FP16 溢出 |
| **Phase 2** | 全 block 归约得到 `sum_sq` | **两级归约**（见 §5.1） |
| **Phase 3** | `normed = x*rstd*(1+scale)+shift` → FP8；写 gate | `rsqrtf`；block 级预计算 `inv_scale`；E4M3 clamp |

---

## 5. 关键技术

### 5.1 两级 Warp/Block 归约（详细）

Phase 1 结束后，256 个线程各自持有一个 **局部部分和** `sum_sq`（各自负责 `D/256` 个元素的 `x²` 累加）。RMSNorm 需要的是 **整行** 的 `Σ x²`，因此 Phase 2 必须做一次 **block 级求和归约**。

本 kernel 采用 **「warp 内 shuffle 归约 + shared memory 跨 warp 归约」** 的两级结构，对应源码：

```cpp
int lane = threadIdx.x % 32, wid = threadIdx.x / 32;

// Level-1: warp 内蝶形归约
for (int o = 16; o > 0; o >>= 1)
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);

// Level-2: 各 warp 代表写 shared memory
if (!lane) shv[wid] = sum_sq;
__syncthreads();

// Level-3: 首 warp 对 8 个 warp 部分和再归约
if (!wid) {
    sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
    for (int o = 16; o > 0; o >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
}
__syncthreads();
if (!threadIdx.x) shv[0] = sum_sq;   // 广播全 block 总和
__syncthreads();
```

---

#### 5.1.1 背景：为什么要分两级？

| 层级 | 参与线程数 | 通信方式 | 原因 |
|------|-----------|----------|------|
| **Warp 内**（32 线程） | 32 | 寄存器 shuffle | 同 warp 线程可在 **一条指令** 内交换寄存器值，**零 shared memory、零 bank conflict** |
| **Warp 间**（8 个 warp） | 256 | `shv[8]` + 首 warp shuffle | shuffle **无法跨 warp**；必须用 shared memory 暂存各 warp 的部分和 |

256 线程 = 8 warps × 32 lanes。若全程用 shared memory 做 tree reduction，需要多次 `__syncthreads()` 且存在 bank conflict 风险；若用 `atomicAdd`，高并发下争用严重。两级方案在 **latency 与资源占用** 之间取得平衡。

---

#### 5.1.2 Warp 与 `__shfl_xor_sync` 是什么？

**Warp** 是 GPU 上 **最小调度单位**：32 个线程以 SIMT 方式 **锁步执行** 同一条指令。同一 warp 内的线程可以通过 **warp shuffle** 指令直接读取彼此 **寄存器** 中的值，无需经过 shared memory 或 global memory。

`__shfl_xor_sync(mask, val, laneMask)` 语义：

- 每个 lane `i` 从 lane `i XOR laneMask` 处读取对方的 `val`
- 本 lane 将读到的值与自身的 `val` 相加（由循环体完成）
- `0xffffffff` 表示 warp 内 32 个 lane 全部参与
- `_sync` 变体保证：**mask 内所有 lane 都到达该指令** 才继续，避免 divergent warp 死锁

---

#### 5.1.3 Warp 内蝶形归约：逐步推演

循环 `for (int o = 16; 8; 4; 2; 1)` 共 **5 步**，每步所有 lane 同时执行一次 shuffle-add。以 8 个 lane 为例（实际 warp 为 32，原理相同）：

**初始**：每个 lane 持有自己的局部 `sum_sq`。

```text
Step o=4:  lane i 与 lane i^4 配对相加
  lane:  0   1   2   3   4   5   6   7
  val:   a   b   c   d   e   f   g   h
         ↓+↓       ↓+↓       ↓+↓       ↓+↓
  结果: a+e  b+f  c+g  d+h  a+e  b+f  c+g  d+h

Step o=2:  lane i 与 lane i^2 配对
Step o=1:  lane i 与 lane i^1 配对（相邻）
```

32-lane 完整 5 步后，**lane 0**（以及同 warp 内所有 lane——shuffle 归约的特性是 **每一步后所有 lane 持有相同聚合值**）持有该 warp 内 32 个线程部分和的总和。

用 lane 配对关系表示（`o=16` 一步）：

```text
lane:  0  1  2  3 ... 15 16 17 ... 31
       ↓+↓  ↓+↓  ↓+↓      ↓+↓  ↓+↓
配对: 0↔16, 1↔17, 2↔18, ... 15↔31
```

这称为 **蝶形（butterfly）** 或 **XOR 归约**：第 `k` 步的配对距离为 `2^k`，与 FFT 蝶形网络同构，保证 log₂(32)=5 步内完成全 warp 归约。

**为何用 XOR 而非 `__shfl_down_sync`？**  
两者均可完成 warp 归约。XOR 模式在 lane 0 和所有 lane 上 **同时得到完整结果**，写法简洁；`shfl_down` 通常只有 lane 0 保留最终值，但本 kernel 在 warp 归约后只让 **lane 0 写 `shv[wid]`**，两种写法此处等价。项目内多个 norm kernel 统一采用 XOR 蝶形写法。

---

#### 5.1.4 Level-2：Warp 间归约

Warp 内归约完成后，每个 warp 的 **lane 0** 执行：

```cpp
if (!lane) shv[wid] = sum_sq;
```

- `wid = threadIdx.x / 32` → warp 编号 0..7
- `shv[8]` 仅存 **8 个 float**（32 bytes），每个 warp 一个部分和
- `__syncthreads()` 保证 8 个写入全部可见

---

#### 5.1.5 Level-3：首 Warp 再归约 + 广播

仅 **warp 0** 的 32 个线程参与（`if (!wid)`）：

```cpp
sum_sq = (lane < 8) ? shv[lane] : 0;   // lane 0..7 加载各 warp 部分和，其余置 0
// 再次 5 步 XOR shuffle → warp 0 内得到全 block 的 sum_sq
```

- 8 个有效值 + 24 个零，shuffle 归约后得到 **8 个部分和之和**
- thread 0 写入 `shv[0]`，第二次 `__syncthreads()` 后 **全部 256 线程** 读取：

```cpp
float rstd = rsqrtf(shv[0] / D + 1e-6f);
```

---

#### 5.1.6 全流程示意（256 线程，D≈1024）

```text
Phase 1  stride 累加
  thread 0:   x[0]² + x[256]² + x[512]² + …
  thread 1:   x[1]² + x[257]² + …
  …
  thread 255: x[255]² + …
        ↓
Level-1  8 次独立的 warp 内 shuffle（各 warp 并行，无 sync）
  warp0: 32 个局部和 → 1 个 warp 部分和
  warp1: …
  …
  warp7: …
        ↓
Level-2  8 次 shared memory 写（lane 0 only）+ __syncthreads__
  shv[0..7] = {W0, W1, …, W7}
        ↓
Level-3  warp0 内 shuffle 归约 8 个数 → shv[0] = W0+…+W7
        ↓
广播     所有线程读 shv[0] → 计算 rstd
        ↓
Phase 3  各线程用同一 rstd 做 RMSNorm + 量化
```

---

#### 5.1.7 好处总结

| 对比方案 | 本方案优势 |
|----------|-----------|
| **全程 shared memory tree reduce** | Warp 内 5 步 shuffle **无 bank conflict、无额外 sync**；仅 warp 间需要 2 次 `__syncthreads()` |
| **`atomicAdd` 到单个地址** | 256 线程争用同一 atomic 槽，serialization 严重；shuffle 是 **O(log N) 步、无争用** |
| **单线程串行累加** | 256 倍并行度浪费，latency 极高 |
| **Global memory 归约** | 延迟比 shared/shuffle 高两个数量级 |

**性能特征**：

- Warp shuffle 延迟约 **~1 cycle 量级**（寄存器旁路），远低于 shared load（~20–30 cycles）或 L2/DRAM
- 5 步 shuffle + 8 元素二次 shuffle：归约部分 **指令数固定**，与 `D` 无关（`D` 只影响 Phase 1 循环次数）
- `shv[8]` 占用极小，不挤压 occupancy

**正确性要点**：

- `_sync` 后缀：warp 内若有分支 divergence，必须用 `_sync` 并保证参与 lane 都执行到 shuffle，否则未定义行为
- 两次 `__syncthreads()`：分隔「写 shv」「读 shv 再归约」「写回 shv[0]」「全线程读 shv[0]」四个阶段，防止数据竞争

---

### 5.2 Launch 配置

```cpp
fused_adarms_fp8_static_fp16_kernel<<<S, 256, 0, stream>>>(...);
```

- **Grid**：`S` 个 block，每个 block 处理一行（一个 action token）
- **Block**：256 线程协作处理 hidden dim `D`
- **Shared memory**：`shv[8]`（32 bytes），静态分配

### 5.3 其他实现细节

| 技术 | 作用 |
|------|------|
| `__restrict__` | 告知编译器指针不别名，便于 load/store 优化 |
| FP32 中间计算 | 归一化与量化全程在 fp32 完成，最后窄化到 fp8/fp16 |
| `fmaxf(*descale_ptr, 1e-12f)` | 防止除零 |
| Gate 落盘 | 有意写 DRAM——C1 到 C4 之间隔多个 kernel，gate 无法只留在寄存器 |

---

## 6. 在 Decoder 流水线中的位置

```text
C1  fused_adarms_fp8_static_fp16
      ├─ 输出 xn_fp8  → C2 fp8_gemm_descale (QKV)
      └─ 输出 gate     → 暂存 bufs['gate']
            ↓
C2–C4: QKV → RoPE → KV cache → Attention → O-proj → fg
            ↓
C4→5: gate_res_adarms_fp8_static_fp16
      x ← x + fg * gate   （使用 C1 写下的 gate）
```

| 分量 | C1 内用途 | 是否落盘 |
|------|-----------|----------|
| scale / shift | 当场用于 norm 后仿射 | 否（结果进入 `xn_fp8`） |
| **gate** | 拷贝到 `gate_out` | **是**（`bufs['gate']`） |

同一 `gate` buffer 在层内会被 **覆盖**：post-attention AdaRMS 产生的新 gate 供 **第二次** gated residual（FFN 后）使用。C1 写的 gate **只服务本层 attn 后的那一次**。

---

## 7. OpenPI ↔ FlashRT 对照

| FlashRT | OpenPI（pi05） |
|---------|----------------|
| `style`（`sa_ptr`） | `precompute_adarms_styles` → `dense(adarms_cond)` 的 scale/shift/gate |
| C1 `fused_adarms_*` | `layer.input_layernorm` + 为 QKV 准备归一化激活 |
| `act_scale_qkv` | 无直接对应（FP8 标定专有，槽位 `l*4+0`） |
| `gate_out` | `input_layernorm` 返回的 gate，用于 `_gated_residual` |

---

## 8. 融合动机

| 动机 | 说明 |
|------|------|
| **减 DRAM 往返** | 避免 `[S,D]` fp16 norm 输出写回再被 QKV GEMM 读取 |
| **减 kernel launch** | Norm + Quant + Gate 拷贝合成一次 launch |
| **对齐 FP8 入口** | QKV Tensor Core GEMM 需要 FP8 输入，在 GEMM 前融合 Norm+Quant |

Gate 落盘是 **有意设计**：与「省掉 norm fp16 中间激活落盘」是两件不同的事。

---

## 9. 相关文件

| 文件 | 角色 |
|------|------|
| `third_party/FlashRT/csrc/kernels/decoder_fused.cu` | Kernel 实现与 host wrapper |
| `third_party/FlashRT/csrc/bindings.cpp` | pybind11 导出 |
| `src/model_optimizer/infer/native/flashrt_decoder/pipeline.py` | C1 调用编排 |
| `src/model_optimizer/infer/native/flashrt_decoder/driver.py` | `DecoderBuffers`（`gate`、`xn_fp8`、`x`） |
| `src/model_optimizer/infer/native/flashrt_decoder/precompute.py` | `precompute_adarms_styles`（`sa/sf/fs`） |
| `docs/optimizer/flashrt/fusion_design.md` | 融合边界与 gate buffer 设计 |
| `docs/optimizer/flashrt/fp8_gemm_descale_fp16.md` | C2 QKV GEMM 与 descale 说明 |
