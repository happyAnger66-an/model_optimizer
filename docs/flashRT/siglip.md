# FlashRT Thor：SigLIP 视觉入口（Patch Embed + 主干）

本文说明 Pi0.5 **Thor Torch 前端**里 SigLIP 的 **patch embedding** 实现（`_patch_embed_ops`）及其与 **27 层 SigLIP 主干**（`siglip_forward`）的衔接。代码路径以 **FlashRT 仓库根目录** 为基准（`flash_rt/` 包）。与整体管线的关系见 [pipeline.md](./pipeline.md)。

---

## 1. 在整条推理中的位置

```text
多视角图像 (NHWC fp16) 写入 _img_buf
  → _patch_embed_ops          # 像素 → token (S_sig × 1152) 写入 _sig_x
  → siglip_forward            # 27 层 Transformer（shared_primitives）
  → postln_project             # PostLN + 投影 + 语言拼接 → encoder_x
```

Patch embed **不包含**在 `siglip_forward` 内部；由前端 **`Pi05TorchFrontendThor._patch_embed_ops`** 显式调用，再进入 **`flash_rt.hardware.thor.shared_primitives.siglip_forward`**。

---

## 2. `_patch_embed_ops`：三步（im2col → GEMM → bias+pos）

实现位置：`flash_rt/frontends/torch/pi05_thor.py` 中 **`_patch_embed_ops(self, stream_int)`**。

### 2.1 维度约定

| 符号 | 含义 | 典型值（Pi0.5） |
|------|------|----------------|
| `num_views` | 相机视角数 | 1 / 2 / 3 |
| `S_sig` | 总 patch 数 = `num_views × 256` | `sig_S` |
| `D_sig` | SigLIP hidden | **1152** |
| 588 | 单 patch 展平长度 **14×14×3** | patch 内 HWC |

- **输入图像缓冲** `_img_buf`：语义 **`(num_views, 224, 224, 3)`** 行主序 **fp16**（`nv × 224 × 224 × 3` 元素）。
- **im2col 输出** `_patches_buf`：`S_sig × 588` 个 fp16。
- **patch 线性输出** `_sig_x`：`torch` 张量 **`(S_sig, D_sig)`**，即 SigLIP **第 0 层输入**。

### 2.2 步 1：`fvk.patch_im2col`

CUDA 实现：`csrc/kernels/patch_embed.cu` 中 **`patch_im2col`**。

- 将 **`(nv, 224, 224, 3)`** 按 SigLIP 的 **16×16 patch 网格**（每视角 **256** 个 patch）展开为 **`(nv×256, 588)`** 矩阵行。
- 每一行对应一个 patch 内 **14×14×3** 像素按约定顺序拉直；与注释中的 reshape / transpose 等价关系一致（与 HF SigLIP patch 划分对齐）。

### 2.3 步 2：`GemmRunner.fp16_nn`（patch → 1152 维）

- **M** = `S_sig`（token 行数）
- **N** = `D_sig`（1152）
- **K** = **588**

即对每个 patch 行向量 **p ∈ R^588** 做 **p · W**，得到 **R^1152**。权重来自 checkpoint 的 **`vision_model.embeddings.patch_embedding.weight`**，在前端构造时做了 **reshape / permute**，使 **W 的列布局与 `patch_im2col` 输出的特征序（HWC im2col order）一致**，再 **`[D_sig, 3,14,14] → … → .T.contiguous()`** 得到供 `fp16_nn` 使用的 **`_pe_w`**（等价 **[588, 1152]** 的 NN GEMM 一侧）。

### 2.4 步 3：`fvk.patch_embed_bias_pos`

同一 C++ 文件中的 **`patch_embed_bias_pos`**：对 **`_sig_x` 原地**做

\[
\text{out}[i,j] \leftarrow \text{out}[i,j] + b[j] + \text{pos\_emb}[(i \bmod 256), j]
\]

- **`bias`**：`_pe_b`，来自 **`patch_embedding.bias`**。
- **`pos_emb`**：`_pos_emb`，来自 **`position_embedding.weight` 的前 256 行**（每视角 256 个空间位置）。
- **`S_per_view` = 256**：多视角时 **行号对 256 取模**，各视角 **共用同一张「图像内 256 位置」位置表**（与 `i % S_per_view` 的 kernel 逻辑一致）。

### 2.5 `stream_int`

传入 **CUDA stream 整型句柄**，使 im2col、GEMM、bias+pos 与后续 **`siglip_forward`**、**CUDA Graph** 捕获使用 **同一 stream**，避免默认流隐式同步破坏捕获。

---

## 3. SigLIP 主干：`siglip_forward`（`shared_primitives.py`）

### 3.1 功能与职责

实现位置：`flash_rt/hardware/thor/shared_primitives.py` 第 **52–147** 行，函数名 **`siglip_forward`**。

- **做什么**：在无框架张量的前提下，走完 **Pi0/Pi05 共用**的 **SigLIP-L 视觉塔**的 **Transformer 主干**——从 **已 patch-embed 的 token 缓冲** **`x ∈ R^{S×D}`**（fp16 **残差流**）出发，重复 **L=27** 层，每层一个 **「Pre-LN 自注意力 + 残差 + Pre-LN FFN + 残差」**，与 HF `SiglipEncoderLayer` / ViT-style block **结构对齐**。
- **不做什么**：**不包含 patch embedding**（Conv/im2col+GEMM+bias/pos）；那一步由前端 **`_patch_embed_ops`** 写好 **`bufs['x']`**（如 `Pi05TorchFrontendThor` 的 **`_sig_x`**）后再调用本函数。
- **量化策略**：大而重的 **GEMM 走 FP8 激活 ×（已量化权重）**，通过 **`gemm.fp8_nn_*`** 与 **`GemmRunner`**（cuBLASLt 一侧）融合 **descale/α**；**注意力**仍在 **FP16** 缓冲区上运行（FMHA）。
- **可插拔注意力**：可选 **`attn=None`**（直接 **`fvk.fmha_strided_full`**），或 **`attn`** 注入 **`AttentionBackend`**，统一 **`attn.run("siglip", …)`**，便于 Thor 上分阶段迁移与对齐测试。

源码 docstring 将整条路径概括为：

> LayerNorm → FP8 → QKV GEMM → FMHA → FP8 → O GEMM+res → LN → FP8 → Up GELU → FP8 → Down+res  

（每层固定；**FMHA + O** 对应一个标准 self-attention 子层，**Up/Down** 对应 MLP/GELU-FFN。）

---

### 3.2 函数签名与依赖

```text
siglip_forward(gemm, fvk, bufs, weights, dims, stream=0, *, attn=None)
```

| 参数 | 含义 |
|------|------|
| **`gemm`** | **`GemmRunner`**：FP8 GEMM（带 bias / GELU / 残差 等 fused 入口）。 |
| **`fvk`** | **`flash_rt_kernels`**：LayerNorm+FP8 量化、FMHA、`quantize_fp8_static_fp16` 等原生指针内核。 |
| **`bufs`** | 各 tensor 的 **device 指针**，见下表。 |
| **`weights`** | 每层一套 **ln/qkv/o/ffn** 权重指针 + **`alpha`** + **`unit_scale`**。 |
| **`dims`** | **`S, D, H, NH, HD, L, num_views, seq_per_view`**。 |
| **`stream`** | CUDA stream 整型句柄。 |
| **`attn`** | 可选 **`ThorFlashAttnBackend`**（或其它实现 **`run("siglip", …)`** 的后端）。 |

**`bufs`（与本函数实际使用的键）**

| Key | 角色 | 典型形状语义（Pi0.5） |
|-----|------|------------------------|
| **`x`** | **残差主缓冲** fp16，`[S, D]` | 入口 = patch embed 输出；层间更新；出口 = SigLIP 最后一层输出。 |
| **`x_fp8`** | LN 后对 **激活** 量化为 FP8 的 **扁平/行布局**缓冲，供 QKV/O/Up 的 GEMM 读入。 |
| **`qkv`** | fp16，`[S, 3D]`，**interleaved**：行内 **Q \| K \| V** 各 **`D`** 维。 |
| **`attn_out`** | fp16，FMHA **输出**，`[S, D]`（与 head 拼接后维度一致）。 |
| **`hidden`** | fp16，FFN **up/GELU 输出**，`[S, H]`。 |
| **`hid_fp8`** | FFN hidden 的 FP8 副本，供 **down** GEMM。 |

注释中还提到 **`scratch (S, max(D,H))`**：由 **外层前端**按需分配（本函数正文未直接使用 `scratch`，与部分导出/校准路径对齐）。

**`weights`**

- **`ln_attn_w/b`、`ln_ffn_w/b`**：每层 **Pre-LN**（注意力前 / FFN 前）的 LN 参数。
- **`qkv_w/b`、`o_w/b`**：**注意力**线性层（合并 QKV **与** O）。
- **`up_w/b`、`down_w/b`**：**FFN**（通常 **hidden H=4304**，**GELU**  fused 在 **`fp8_nn_gelu_bias`**）。
- **`alpha`**：**host float 数组**，长度 **`L × 4`**，每层 **`a_qkv, a_o, a_up, a_down`** —— 写进 FP8 GEMM 的 **标量缩放**（与校准/静态量化尺度一致）。
- **`unit_scale`**：**device fp32**，`quantize_fp8_static_fp16` 使用；注释说明 LN 输出已归一化，此处可用 **尺度 1.0** 的把输出 cast 进 FP8 供下一 GEMM。

**`dims`**

- **`S`**：总视觉 token = **`num_views × seq_per_view`**（一般为 **`nv × 256`**）。
- **`D`**：SigLIP **hidden**，Pi0.5 为 **1152**。
- **`H`**：FFN intermediate，Pi0.5 为 **4304**。
- **`NH`、`HD`**：**头数 × head_dim**，且 **`NH × HD == D`**（此处 **NH=16, HD=72**）。
- **`L`**：层数 **27**。
- **`num_views`、`seq_per_view`**：供 FMHA **按视角独立**：**batch 轴 = nv**，每条序列长度 **256**。

---

### 3.3 单层内数据流（第 `l` 层，`l = 0 … L-1`）

对每个 **`l`** 从 **`alpha`** 取四个标量后直接执行固定序列。

**Attention 子层**

1. **`fvk.layer_norm_fp8`**：`x →` 写入 **`x_fp8`**（附带 **LN 权重**），等价 **Pre-LN + 量化到 FP8**。
2. **`gemm.fp8_nn_bias`**：`x_fp8 @ qkv_w + qkv_b → qkv`，形状 **`[S, 3D]`**。
3. **Self-attention**（二选一）：
   - **`attn.run("siglip", 0, q_seq=spv, …)`**：`layer_idx` 对 SigLIP 固定为 **`0`**（所有 **27** 层复用同一套 **slot**：都是 **交错 qkv.buffer + attn_out**，层间只靠 **权重 `[l]`** 替换）。
   - **`fvk.fmha_strided_full`**：**`Q=qkv`、`K=qkv+D*2`、`V=qkv+4D`（字节偏移)**，**stride = `3D`**（以 **fp16 元素语义**对齐行内 Q/K/V 步长）。**`nv`** 视作 batch， **`spv×spv`** 为 **per-view QK 方阵**，即 **摄像头之间互不 attend**。
   - softmax 缩放 **隐式或使用 `HD`**：源码里 **`scale = 1/sqrt(HD)`** 若走某些路径；`fmha_strided_full` 接口内部会使用等价缩放（与标准 scaled dot-product attention 一致）。
4. **`fvk.quantize_fp8_static_fp16`**：**`attn_out → x_fp8`**，用 **`unit_scale`**。
5. **`gemm.fp8_nn_bias_res`**：**`x += α_o · ( x_fp8 @ o_w + o_b )`**，**原地残差回到 `x`**。

**FFN 子层**

6. **`fvk.layer_norm_fp8`**：再次 **Pre-LN** 写 **`x_fp8`**（**`ln_ffn`**）。
7. **`gemm.fp8_nn_gelu_bias`**：**Up 投影 + GELU + bias → `hidden`**，带 **`α_up`**。
8. **`quantize_fp8_static_fp16`**：**`hidden → hid_fp8`**。
9. **`gemm.fp8_nn_bias_res`**：**`x += α_down · ( hid_fp8 @ down_w + down_b )`**。

循环结束后：**`bufs['x']`** 即为 **SigLIP 顶层输出**，再交给 **`postln_project`**（Post-LN + 多模态投影 + 语言槽），见本文第 **1** 节与 **`postln_project`**。

---

### 3.4 多头与「视角独立」注意的语义

- **QKV / O**：与 HF 一致，**NH=16**，**HD=72**，嵌入维 **1152**。
- **不按「整段长度 S」做一次全局 softmax**，而是 **`nv` 个小图**，每个 **`256×256`**：即 **视角内** patches 互为 context，**跨视角无 attention**——符合「多相机各看一张 fixed grid」的工程假设，也使得 **VRAM 与算力**按视角线性缩放、且与 **FLASH strided FMHA** 适配。

---

### 3.5 与 Hugging Face `modeling_siglip` 的对照（概念）

- **Pre-LN + Attention + residual + Pre-LN FFN + residual**：与同构 **ViT Encoder block** 一致。
- HF 常为 **Separate QKV / O Linear**；本处 **仍为同一数学**，只是把 **GEMM+Fused**、**FMHA** 和 **FP8** 路径写死在指针层。
- **Patch + position**：在 **`SiglipVisionEmbeddings`**；本函数仅从 **等价于 `.forward` 之后 `embeddings`** 的张量 **`x`** 开始。

---

### 3.6 谁在调用、如何接 `bufs['x']`

- **`Pi05TorchFrontendThor`**：`_sig_bufs['x'] = _sig_x.data_ptr()`，**`_patch_embed_ops`** 写好 **`_sig_x`**，再 **`siglip_forward(..., attn=self._attn)`**。
- **Pi0 / GROOT**：同样复用 **`siglip_forward`**（见 **`shared_primitives.py` 头部说明**）。

---

## 4. CUDA Graph 与 warmup 注意点

`_capture_siglip_graph`（`pi05_thor.py`）将 **`_patch_embed_ops` + `siglip_forward` + `_postln_project_ops`** 打入 **同一张 CUDA Graph**。

Warmup 流程中会先跑 **`_patch_embed_ops`**，再对 **`_sig_x` 执行 `zero_()`**，再进入 **`siglip_forward`**。目的：**patch 路径仍被捕获/预热**，但 **进入 27 层前 token 置零**，避免「全零图像 + bias 累乘」在深层产生 **inf**（源码注释已说明）。生产路径若同样依赖「先 embed 再清零或再写真实特征」，需与捕获策略保持一致。

---

## 5. 源码索引

| 路径 | 内容 |
|------|------|
| `flash_rt/frontends/torch/pi05_thor.py` | **`_patch_embed_ops`**；`_pe_w` / `_pe_b` / `_pos_emb` / `_img_buf` / `_patches_buf` / `_sig_x` 分配与 `_capture_siglip_graph` |
| `csrc/kernels/patch_embed.cu` | **`patch_im2col`**、**`patch_embed_bias_pos`** 的 GPU kernel 与形状注释 |
| `flash_rt/hardware/thor/shared_primitives.py` | **`siglip_forward`**（27 层 + FMHA / backend） |
| `flash_rt/hardware/thor/attn_backend.py` | Thor 上 **`attn.run("siglip", …)`** 与 `fmha_strided_full` 的对应关系 |

---

*若与 FlashRT 主分支实现有出入，以仓库当前源码为准。*
