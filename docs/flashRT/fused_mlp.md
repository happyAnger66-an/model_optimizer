# FlashRT Thor：FFN 中 Gate+Up 合并与 GEGLU 融合

本文总结 **FlashRT** 在 **Pi0.5 / Paligemma 类编码器**（Thor 路径）里，对 **GEGLU 式 FFN** 在 `encoder_forward` 中相邻两步的数学含义与合并优化要点。代码以 **FlashRT 主仓**为准；若 `model_optimizer` 通过子模块使用，一般对应 `third_party/FlashRT/` 下同路径。

**入口**：`flash_rt/hardware/thor/shared_primitives.py` 中 `encoder_forward`（约第 287–296 行）。

**相关 CUDA**：

- FP8 GEMM：`csrc/gemm/cutlass_sm100.cu`（`cutlass_fp8_t1` → `sm100_t1::Gemm`，FP8 入、FP16 出）。
- GEGLU + 量化：`csrc/kernels/activation.cu`（`gate_silu_mul_merged_fp8_kernel_fp16` 等；函数名历史上有 `silu`，实现为 **tanh 近似 GELU**）。

---

## 1. 参考：GEGLU FFN（与实现对齐）

设序列长度方向批量为 \(S\)（代码里 `Se`），隐藏维 \(H\)，模型维 \(D\)：

\[
g = x W_g,\quad u = x W_u \quad (x \in \mathbb{R}^{S\times D},\; W_g, W_u \in \mathbb{R}^{D\times H})
\]

\[
h = \mathrm{GELU}(g) \odot u
\]

\[
y = h W_{\mathrm{down}}
\]

FlashRT 中 \(\mathrm{GELU}\) 为 **tanh 近似**（与 `activation.cu` 中常数 `1.595769...`、`0.044715` 一致），与 Gemma / Paligemma 常用近似一致。

---

## 2. 第一步：合并 Gate+Up 线性层 — `cutlass_fp8_t1`

**调用示例**（`encoder_forward` 内，非最后一层）：

```python
fvk.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                   Se, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)
```

**CUTLASS 问题规模**：`M = Se`，`N = H * 2`，`K = D`。即对每个 token，等价于 **`[1×D] × [D×2H] → [1×2H]`**，整块为 `Se` 行。

| 参数 / 缓冲 | 含义 |
|---------------|------|
| `x_fp8` | `[Se, D]`，残差 + RMSNorm 后再 **FP8 量化** 的激活。 |
| `gate_w[l]` | **FP8** 权重；逻辑上将 **`W_g` 与 `W_u` 在输出维上拼成宽度 `2H`**，与单次列方向 `2H` 的线性一致。 |
| `gate` | `[Se, 2H]`，**FP16** 输出；**前半 `H` 列为 gate 分支 logits，后半 `H` 列为 up 分支 logits**，此时 **尚未** 做 GELU。 |
| `alpha_host[l*4+2]` | 该量化点的 **静态 descale**：**`act_scale × weight_scale`**，在标定阶段预计算，融进 GEMM。 |

**相对「两次独立 GEMM」的优化**：

| 未合并 | 合并后 |
|--------|--------|
| 两次 `x_fp8 @ W_g`、`x_fp8 @ W_u`：两次读激活、两次 launch | **一次** `cutlass_fp8_t1`：只读一遍 `x_fp8`，一次大 GEMM 写出完整 `[g_\text{pre} \| u_\text{pre}]`。 |
| 两次中间写 | **一次** 写满 `gate`（布局见下节）。 |

**Tile 选型**：`cutlass_fp8_t1` 使用 `sm100_t1::Gemm`（见 `cutlass_sm100.cu` 头注释：**T1 用于 Gate+Up、利于 L2**），与 QKV/O-proj 用的 `cutlass_fp8_sq`、Down 用的 `cutlass_fp8_wide` 区分开，属于 **性能调参**，不改变上述数学关系。

---

## 3. 第二步：GEGLU + FP8 量化 — `gate_geglu_merged_fp8_fp16`

**调用示例**：

```python
fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H, as_d, stream)
```

**布局约定**（与 `activation.cu` 注释一致）：

- 输入 `gate`：`merged[s, 0..H-1]` = **gate 分支**，`merged[s, H..2H-1]` = **up 分支**。

对每个 `(s, h)`，\(h = 0..H-1\)：

1. 读 \(g = \text{gate}[s, h]\)，\(u = \text{gate}[s, H+h]\)。
2. 计算 \(\mathrm{GELU}(g) \times u\)（tanh 近似 GELU）。
3. 用设备标量 **`as_d`**（`d_scale`）将结果 **量化到 FP8 E4M3**，写入 **`hid_fp8[s, h]`**，形状 **`[Se, H]`**。

内核中将 **读 merged、GELU、乘、clamp、写 FP8** 放在同一线程路径；FP16 路径采用 **每线程 4 元素** 的向量化实现（`gate_silu_mul_merged_fp8_kernel_fp16`）。

**相对「多 kernel 链」的优化**：

| 朴素拆法 | 合并后 |
|----------|--------|
| 先写 `GELU(g)` 到 `[S,H]` FP16，再与 `u` 逐元相乘，再单独 `quantize_fp8` | **单 kernel**：从同一缓冲的两段读 `g,u`，算 **`GELU(g)×u`**，**直接** 得到 `hid_fp8`，减少中间张量与全局内存往返。 |
| gate/up 分两个 buffer | **merged `[g‖u]`**：同一行连续存储，便于 **half2** 向量化加载。 |

Python 绑定名含 `geglu`；底层 C 符号仍为 `gate_silu_mul_merged_fp8_fp16`（历史命名）。

---

## 4. 与整段 FFN 的衔接

在 `encoder_forward` 中，上述两步之后通常还有 **Down 投影**（`cutlass_fp8_wide`：`hid_fp8` × `down_w` → FP16 等），再经残差 + RMSNorm 进入下一层。整体数据流可概括为：

1. **`cutlass_fp8_t1`**：`x_fp8` → **`gate` `[Se, 2H]` FP16**（合并线性 Gate+Up）。
2. **`gate_geglu_merged_fp8_fp16`**：`gate` → **`hid_fp8` `[Se, H]` FP8**（GEGLU + 为 Down 准备的量化激活）。
3. **`cutlass_fp8_wide`**：`hid_fp8` → Down 输出（再接残差等）。

---

## 5. 与「拆分 Gate / Up」路径的对比（概念）

FlashRT 中另有 **`silu_mul_split_fp8_fp16`** 等路径：对应 **两次 GEMM** 分别产出 gate/up，再融合激活（见 `activation.cuh` 注释：split 用于 **L2** 等不同取舍）。**当前 `shared_primitives.py` 编码器片段**采用的是 **「一次合并 GEMM + merged GEGLU 内核」**，与 split 路径是 **不同工程折中**，数学上在「两路线性再 GEGLU」意义上仍应对齐同一参考公式。

---

## 6. 延伸阅读（FlashRT 仓库内）

| 主题 | 路径 |
|------|------|
| Kernel 清单与命名 | `docs/kernel_fusion.md`（`cutlass_fp8_t1`、`gate_geglu_merged_fp8_fp16` 条目） |
| 静态 FP8 scale / alpha | `docs/calibration.md`（与 `alpha_host`、`act_scales` 配合） |

本文档侧重 **Thor `encoder_forward` 中 FFN 段的两步融合**；TensorRT-LLM 侧的 **`fuse_gate_mlp` / FusedGatedMLP** 见 `docs/optimizer/fused_mlp.md`。
