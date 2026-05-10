# `rms_norm_fp8_noweight_fp16`：计算流程与优化

源码：`third_party/FlashRT/csrc/kernels/norm.cu`（`rms_norm_fp8_noweight_fp16` 668–672 行 + 设备核 `rms_norm_fp8_noweight_kernel` 603–666 行）。Python 侧典型调用见 `flash_rt/hardware/thor/shared_primitives.py` 中 `encoder_forward` 每层第一步。

---

## 1. Host 封装

```668:672:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/norm.cu
void rms_norm_fp8_noweight_fp16(const __half *x, __nv_fp8_e4m3 *out,
                                int seq_len, int dim, const float *d_scale,
                                cudaStream_t stream) {
  rms_norm_fp8_noweight_kernel<<<seq_len, 256, 0, stream>>>(x, out, seq_len,
                                                            dim, d_scale);
}
```

- **Grid**：**`seq_len` 个 block**，**每个 token（行）一个 block**（`blockIdx.x == r`）。
- **Block**：**256 线程**。
- **动态共享内存**：**0** 字节（`<<<..., 0, stream>>>`）。
- **`dim`** 在核内记为 **`C`**；实现按 **`C ≤ 2048`** 设计（见宏）。

---

## 2. 设备核：数学计算（逐行 RMS → 乘 scale → FP8）

对 **固定行 `r`**：

1. **读入一行** `in[r*C : (r+1)*C)`（FP16），视为 **`__half2*`**，即 **`C/2` 个 half2**。
2. **每个线程**在负责的一段 **`c2` 索引**上累加 **部分平方和 `ssq`**，并把用到的 **`__half2 → float`** 放进 **寄存器数组 `cache[8]`**。
3. **全 block 规约**得到 **`sh[0] = Σ_j x_j²`**（先 warp 内 **`__shfl_xor_sync`**，再 **8 个 warp** 用 **`sh[wid]` + 再 shuffle** 合成）。
4. **标量**  
   `scale = rsqrt( sh[0] / C + 1e-6 ) / max(*descale_ptr, 1e-12)`  
   使用 **`__frsqrt_rn`**；**`descale_ptr`** 即标定用的 **`d_scale`**（设备上单个 `float32`）。
5. **第二遍循环**：用 **`cache`** 中已转好的 float，**不再从 global 读 `in`**，算  
   `y = clamp(cache * scale, ±448)` → **`__nv_fp8_e4m3`**，并用 **`uint16_t*` 一次写 2 个 FP8**。

**无 γ/β**：没有 RMS 可学习仿射；与头文件注释一致，**norm 权重已 bake 进后续 GEMM**。

---

## 3. 线程如何覆盖一整行（`C=2048`）

```599:613:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/norm.cu
#define RMS_NW_THREADS 256
#define RMS_NW_D_MAX 2048
#define RMS_NW_ELEMS_PER_THREAD (RMS_NW_D_MAX / RMS_NW_THREADS) // 8
...
  const __half2 *row2 = reinterpret_cast<const __half2 *>(row);
  int C2 = C / 2;
```

- **`C2 = C/2`**：half2 个数 = **1024**（当 `C=2048`）。
- 外层 **`it = 0 .. 3`**（**`RMS_NW_ELEMS_PER_THREAD/2 = 4`**），内层  
  **`c2 = threadIdx.x + it * blockDim.x`**  
  → 每线程处理 **4 个 `c2`**，共 **256×4 = 1024** 个 half2，铺满一行。

**`cache[8]`** 与每线程 **最多 8 个 float** 对应。

---

## 4. `ssq` 规约路径

```632:648:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/csrc/kernels/norm.cu
  __shared__ float sh[16];
  int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    ssq += __shfl_xor_sync(0xffffffff, ssq, o);
  if (!lane)
    sh[wid] = ssq;
  __syncthreads();
  if (!wid) {
    ssq = (lane < (blockDim.x / 32)) ? sh[lane] : 0;
    for (int o = 16; o > 0; o >>= 1)
      ssq += __shfl_xor_sync(0xffffffff, ssq, o);
  }
  __syncthreads();
  if (!threadIdx.x)
    sh[0] = ssq;
  __syncthreads();
```

- **Warp 内**：`ssq` 经 **XOR shuffle** 在 32 lane 上规约。
- **Warp 间**：lane0 写入 **`sh[wid]`**（**8 个 warp**；**`sh[16]`** 足够）。
- **Warp0** 再读 **`sh[lane]`** shuffle 规约，**thread0** 写入 **`sh[0]`** 广播。

**不用 `atomicAdd` 浮点原子**。

---

## 5. 优化手段归纳

| 手段 | 作用 |
|------|------|
| **每行一 block** | 行与行独立，**`seq_len` 维高并行**；规约仅在 **256 线程**内完成。 |
| **`__half2` 向量读** | 相对逐 `half` 读，**带宽更好**，指令更少。 |
| **寄存器 `cache[8]`** | 第一遍读入并转 float 后，**写 FP8 不再读 global `in`**，省 **约一半 global load**。 |
| **`#pragma unroll`** | 利于 **ILP** 与分支可预测性。 |
| **`__shfl_xor_sync` 规约** | Warp 内高效求和；跨 warp 用 **极小 shared** 中转。 |
| **`__frsqrt_rn`** | **融合 rsqrt**，替代 `sqrt` + 除法。 |
| **FP8 成对 `uint16_t` 写** | 两个 E4M3 **16-bit 合并写回**，减少 store。 |
| **`fminf/fmaxf` ±448 clamp** | 与 **E4M3** 可表示范围一致。 |
| **固定 `D_MAX=2048`** | **`cache` 与循环次数编译期固定**；**`dim` 若大于 2048 会算不全**（本产品线 encoder **`D=2048`** 对齐该假设）。 |
| **动态 shared = 0** | Launch 简单，减少占用率与配置复杂度。 |

---

## 6. 与 `encoder_forward` 的衔接（索引）

`shared_primitives.encoder_forward` 每层先：

```252:257:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/hardware/thor/shared_primitives.py
        fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, D, as_qkv, stream)
        fvk.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv, ...)
```

即：**本核输出 `x_fp8` → 下一 QKV FP8 GEMM**；**`as_qkv`** 为 **`act_scales` 设备缓冲**上第 **`l*4+0`** 个 `float32` 的地址（字节偏移 **`(l*4+0)*4`**）。

更完整的 **`act_scales` / `alpha_host` 分工** 见同目录 **`rms_norm_fp8.md`**。

---

## 7. 一句话

**对每一行做 RMS 逆缩放并除以标定 `d_scale`，饱和写入 E4M3**；通过 **half2 读、寄存器缓存、shuffle+小 shared 规约、frsqrt、FP8 打包写、按行并行与展开**，在 **`C=2048`** 假设下做成 **高带宽、低原子、少重复读** 的生产核。
