# C2b：`qkv_split_rope_kvcache_fp16` 算子说明

> **实现**：`third_party/FlashRT/csrc/kernels/rope.cu::qkv_split_rope_kvcache_fp16_kernel`  
> **Host 入口**：`qkv_split_rope_kvcache_fp16`（`<<<blocks, 256>>>`）  
> **Python 绑定**：`fvk.qkv_split_rope_kvcache_fp16`（`flashrt_decoder/kernels.py` 加载）  
> **编排位置**：`flashrt_decoder/pipeline.py::decoder_forward` 每层 C2b（紧接 C2 QKV GEMM 之后）  
> **RoPE 预计算**：[`docs/optimizer/flashrt/dec_rope.md`](../../optimizer/flashrt/dec_rope.md)（`build_dec_rope` → `weights['rope']`）  
> **上游**：[`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md)（C2）  
> **融合设计**：[`docs/optimizer/flashrt/fusion_design.md`](../../optimizer/flashrt/fusion_design.md) §2

---

## 1. 功能概述

`qkv_split_rope_kvcache_fp16` 是 **C2b 融合 kernel**，在一次 GPU launch 中完成 OpenPI 每层 self-attention **前半段**的三件事：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | **Split** | 将合并的 `qkv [S, 2560]` 拆成 Q / K / V |
| 2 | **RoPE** | 对 Q、K 施加旋转位置编码（**V 不做 RoPE**） |
| 3 | **写 KV cache** | RoPE 后的 K 与原始 V 写入 `Kc` / `Vc` 的 **suffix 区** |

PyTorch/OpenPI 中这些步骤分散在 Gemma Expert 的 `self_attn` 内；FlashRT 合并为单 kernel，减少 launch 次数与 DRAM 往返。

---

## 2. 在 OpenPI Denoise 中的对应关系

### 2.1 Denoise 调用链

OpenPI `pi0_pytorch.py::denoise_step` 每扩散步调用 Expert forward（仅 suffix 分支）：

```python
# denoise_step 内
outputs_embeds, _ = self.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,          # suffix 从 enc_seq 起算
    past_key_values=past_key_values,    # prefix KV（vit+llm 阶段产出）
    inputs_embeds=[None, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],
)
```

每层 Expert（`gemma_pytorch.py`，`i=1`）在 attention 之前：

```python
query_state = layer.self_attn.q_proj(hidden_states).view(...).transpose(1, 2)
key_state   = layer.self_attn.k_proj(hidden_states).view(...).transpose(1, 2)
value_state = layer.self_attn.v_proj(hidden_states).view(...).transpose(1, 2)

cos, sin = rotary_emb(..., position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

att_output, _ = eager_attention_forward(...)  # suffix Q attend to prefix+suffix KV
```

### 2.2 FlashRT ↔ OpenPI 对照

| OpenPI（denoise 单层） | FlashRT |
|------------------------|---------|
| `q/k/v_proj`（input_layernorm 后） | C2 `fp8_gemm_descale_fp16` → 合并 `qkv` |
| reshape + split Q/K/V | **C2b** split |
| `apply_rotary_pos_emb`（全局位置 `enc_seq..enc_seq+S`） | **C2b** + 预计算 `rope`（`build_dec_rope`） |
| suffix K/V 供 attention 使用 | **C2b** 写入 `Kc/Vc[:, enc_seq:, :]` |
| `eager_attention_forward` | C3 `attention_qkv_fp16` |

### 2.3 位置编码对齐

OpenPI denoise 中 suffix 的 `position_ids` 从 **prefix 长度** 起算，而非从 0 重新编号：

```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

FlashRT `build_dec_rope(enc_seq, Sa)` 预计算 `rope[s, :]`，切片全局位置 `[enc_seq, enc_seq+Sa)` 的 cos/sin，与 OpenPI `position_ids` 语义一致。详见 [`dec_rope.md`](../../optimizer/flashrt/dec_rope.md)。

---

## 3. 混合 KV 路径（Prefix + Suffix）

Pi0.5 denoise 采用 **TRT prefix + FlashRT suffix** 的混合 KV 设计：

```text
Prefix 阶段（TRT vit+llm）:
  图像+语言 → past_key_values，长度 enc_seq
  → fill_prefix_kv_from_trt 写入 Kc/Vc[:, :enc_seq, :]

Denoise 阶段（FlashRT，10 步 × 18 层）:
  每步、每层:
    C2   产出 suffix raw qkv（尚未 RoPE）
    C2b  RoPE(Q,K) + 写 suffix KV → Kc/Vc[:, enc_seq:enc_seq+S, :]
    C3   suffix Q 对 [prefix + suffix] 全序列 attention（total_keys = enc_seq + S）
```

| 区域 | 来源 | RoPE |
|------|------|------|
| `Kc/Vc[:, :enc_seq, :]` | TRT prefix（`set_prefix_kv`） | prefix 侧已在 TRT 路径完成 |
| `Kc/Vc[:, enc_seq:, :]` | C2b 每层写入 | C2b 对 suffix Q/K 施加 `build_dec_rope` 表 |

OpenPI `use_cache=False` 的 denoise 每步重算 suffix QKV；FlashRT 同样每步每层 C2+C2b 重算 suffix，prefix 区复用。

---

## 4. 调用形式与参数

### 4.1 Pipeline 调用

```python
kv_offset = l * total_keys * HD + enc_seq * HD
fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                S, Q_dim, K_dim, HD, 2560,
                                kv_offset, HD, stream)
```

### 4.2 参数说明

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `qkv` | `[S, 2560]` fp16 | C2 输出的合并 Q+K+V（raw，无 RoPE） |
| `rope` | `[S, 2560]` fp16 | suffix RoPE 表，交错 `[cos0,sin0,cos1,sin1,...]` |
| `attn_out` | `[S×NH, HD]` fp16 | **输出 Q**（RoPE 后），供 C3 attention |
| `Kc` / `Vc` | `[layers, total_keys, HD]` | 全层 KV cache；prefix 已填，suffix 由本 kernel 写 |
| `S` | 10 | action token 数（`action_horizon`） |
| `Q_dim` | 2048 = NH×HD | 8 query heads × 256 |
| `K_dim` | 256 = HD | GQA：1 个 KV head |
| `HD` | 256 | head 维度 |
| `2560` | `qkv_stride` | `Q_dim + K_dim + V_dim` |
| `kv_offset` | `l×total_keys×HD + enc_seq×HD` | 第 `l` 层 suffix 起点在 flat cache 中的元素偏移 |
| `HD` | `kc_stride` | cache 中相邻 token 的步长 |

Pi0.5 典型维度：`NH=8`，`HD=256`，`enc_seq` 随 prompt 变化（如 libero ~818），`total_keys = enc_seq + S`。

### 4.3 `kv_offset` 计算

```text
Kc/Vc 布局: [layer][token_pos][head_dim]
  token_pos ∈ [0, total_keys)
    [0, enc_seq)           = prefix（TRT 填入）
    [enc_seq, total_keys)  = suffix（C2b 写入）

layer l 的 suffix 起点（元素偏移）:
  kv_offset = l * total_keys * HD + enc_seq * HD
```

`kv_offset` 中的 `enc_seq` 必须与 `build_dec_rope` 切片起点 **一致**。

---

## 5. Kernel 内部逻辑

### 5.1 三路分支

对每个 `(s, c)`，`c = idx % qkv_stride`，`s = idx / qkv_stride`：

```text
c ∈ [0, Q_dim)            → Q 路
c ∈ [Q_dim, Q_dim+K_dim)  → K 路
c ∈ [Q_dim+K_dim, 2560)   → V 路
```

| 路径 | 操作 | 输出 |
|------|------|------|
| **Q** | 读 qkv 维度对 `(x0,x1)`，查 `rope[s]`，RoPE 旋转 | `attn_out[s*Q_dim + c]` |
| **K** | 同上 RoPE | `Kc[kc_offset + s*kc_stride + k_col]` |
| **V** | 直接拷贝（无 RoPE） | `Vc[kc_offset + s*kc_stride + v_col]` |

### 5.2 RoPE 公式

与 Gemma `apply_rotary_pos_emb` 一致，对维度对 `(x0, x1)` 与 `(cos, sin)`：

```
偶数维: out = x0 * cos - x1 * sin
奇数维: out = x1 * cos + x0 * sin
```

Kernel 从 `rope[s*HD + pair*2]` 读 cos，`rope[s*HD + pair*2 + 1]` 读 sin。

### 5.3 Launch 配置

```cpp
blocks = ceil(S * qkv_stride / 256)
threads = 256
```

每个线程处理 `qkv` 的一个元素，按列索引 `c` 分支到 Q/K/V 三路。

---

## 6. 单步单层时序

```text
C1  fused_adarms_fp8_static_fp16     → xn_fp8
C2  fp8_gemm_descale_fp16           → qkv [S, 2560]  （raw Q/K/V，无 RoPE）
C2b qkv_split_rope_kvcache_fp16     ← 本算子
      ├─ attn_out                   → RoPE 后的 Q，给 C3
      └─ Kc/Vc suffix 区            → RoPE 后的 K + 原始 V
C3  attention_qkv_fp16
      Q   = attn_out   [S×NH, HD]
      K/V = Kc/Vc      [total_keys, HD]  （prefix + 刚写入的 suffix）
C4  O-proj FP8 GEMM ...
```

**融合动机**：PyTorch 中 split、RoPE、写 cache 为 3 个独立 op；FlashRT 合成 1 个 kernel，`qkv` 只读一次 DRAM，Q/K/V 分流写出。

---

## 7. GQA 与 qkv 布局

Pi0.5 Expert 使用 **Grouped Query Attention**：

| 分量 | head 数 | 每 token 维度 |
|------|---------|---------------|
| Q | 8（NH） | NH × HD = 2048 |
| K | 1 | HD = 256 |
| V | 1 | HD = 256 |
| **合计** | | **2560** |

`qkv` 行内布局：`[Q_0..Q_2047 | K_0..K_255 | V_0..V_255]`。

C2 合并 QKV 权重 `qw [D, 2560]` 与此 layout 对应（`weights.py` repack + `interleave_qk`）。

---

## 8. 精度与契约要点

| 要点 | 说明 |
|------|------|
| **`rope` 与 `enc_seq` 对齐** | prompt 变长 → `FlashRtDecoderBackend.setup_prompt(enc_seq)` 必须重建 RoPE 表 |
| **prefix / suffix 分工** | prefix K/V 来自 TRT；C2b 只处理 suffix Q/K 的 RoPE 并写 suffix K/V |
| **每 denoise 步重算** | 与 OpenPI `use_cache=False` 一致；suffix QKV 每步每层由 C2+C2b 重算 |
| **RoPE 表非训练权重** | 仅由 `enc_seq`、`Sa`、`head_dim` 决定；与 checkpoint 无关 |
| **混合路径验证** | TRT prefix KV + FlashRT suffix RoPE 布局须与 PyTorch 一致，见 `native_decoder_implementation_todo.md` §8.2 |

---

## 9. 相关文件

| 文件 | 角色 |
|------|------|
| `third_party/FlashRT/csrc/kernels/rope.cu` | Kernel 实现 |
| `third_party/FlashRT/csrc/kernels/rope.cuh` | 声明 |
| `third_party/FlashRT/csrc/bindings.cpp` | pybind11 导出 |
| `src/model_optimizer/infer/native/flashrt_decoder/pipeline.py` | C2b 调用与 `kv_offset` |
| `src/model_optimizer/infer/native/flashrt_decoder/precompute.py` | `build_dec_rope` |
| `src/model_optimizer/infer/native/flashrt_decoder/backend.py` | `setup_prompt` 挂 `weights['rope']` |
| `src/model_optimizer/infer/native/flashrt_decoder/driver.py` | `Kc/Vc` buffer、`fill_prefix_kv_from_trt` |
| `third_party/openpi/.../pi0_pytorch.py` | `denoise_step` |
| `third_party/openpi/.../gemma_pytorch.py` | Expert 层 QKV + RoPE + Attention |
| `docs/optimizer/flashrt/dec_rope.md` | RoPE 表预计算 |
| `docs/flashRT/denoise/fp8_gemm_descale_fp16.md` | C2 上游 QKV GEMM |
