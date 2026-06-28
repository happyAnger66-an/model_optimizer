# Pi0.5 推理优化总结（`tensorrt_native_denoise.yaml`）

> 对照原始 PyTorch 流程：`third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py`  
> 示例配置：`config/webui_configs/tensorrt_native_denoise.yaml`

---

## 一、整体架构：从「单一 PyTorch 图」到「分阶段异构后端」

原始 `pi0_pytorch.py` 的 `sample_actions` 是一条**纯 PyTorch / HF transformers** 的同构流程：

```python
# pi0_pytorch.py::sample_actions（简化）
prefix_embs, ... = self.embed_prefix(...)
_, past_key_values = self.paligemma_with_expert.forward(..., use_cache=True)  # ① prefill
while time >= -dt / 2:                                                          # ② 10 步去噪循环
    v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
    x_t = x_t + dt * v_t
return x_t
```

`tensorrt_native_denoise.yaml` 把同一个推理图**按阶段拆成 3 种后端**（FlashRT 文档中的 per-stage backend matrix）：

| 阶段 | 原始 PyTorch | 本配置后端 | 配置项 |
|------|-------------|-----------|--------|
| ViT / SigLIP（视觉编码） | HF SigLIP，逐视角 | **TensorRT engine**（FP8 + 多视角 batch） | `vit_engine: vit_fp8_batch.engine` |
| LLM prefill（PaliGemma） | HF eager attention | **TensorRT engine**（FP8 + KV-only） | `llm_engine: llm_kv_fp8.engine` |
| denoise（18 层 × 10 步） | `denoise_step` 逐步 PyTorch | **仓内 FlashRT decoder**（整循环融合 kernel） | `native_flashrt_decoder: true` |

关键配置：

```yaml
inference_mode: tensorrt          # vit/llm 走 TRT
native_overlay_on_tensorrt: true  # 在 TRT 之上叠加 native，仅覆盖 denoise
native_enable_denoise: true
denoise_engine: ""                # denoise 不走 TRT engine，交给 FlashRT
```

**核心思路**：让每个阶段用各自最优的后端——大 batch 的 SigLIP/LLM 交给 TensorRT 的 Myelin 全局融合；launch-bound 的 denoise（S=10，小 batch）交给手写融合 kernel。

---

## 二、TensorRT Engine 替换（ViT + LLM）

### 2.1 ViT：编译期图优化 + 多视角 batching

**原始**：`embed_prefix` 里逐视角串行调用 SigLIP（N 视角 = N 次前向）。

**本配置**：`vit_batch_views: true` + `vit_fp8_batch.engine`

- 多视角堆成 batch 维，**一次 engine 调用**处理所有相机视角
- TensorRT 编译期完成 LayerNorm/GELU/bias 作为 GEMM epilogue 的融合、layout 选择、tactic autotune
- engine 自身为 **FP8**（权重 + 激活量化）

#### 2.1.1 背景

LIBERO 等任务通常有多个相机（如 base + wrist，2 路）。每路 `224×224` 图像经 SigLIP 变为 256 个 image token，再与 language token 拼成 prefix。

原始 OpenPI `embed_prefix`（`pi0_pytorch.py`）：

```python
for img, img_mask in zip(images, img_masks, strict=True):
    img_emb = self.paligemma_with_expert.embed_image(img)  # 每视角 1 次 get_image_features
    embs.append(img_emb)
```

**N 视角 = N 次 ViT/TRT 前向**（N 次 kernel launch 序列）。

#### 2.1.2 优化原理

SigLIP 对**每张图独立编码**：self-attention 只在单张图内部 256 个 patch token 之间做，**batch 维上不同图片互不可见**。因此可把 N 张图在 batch 维拼成 `[N, 3, 224, 224]`，一次过 ViT，输出再按视角切回——与 N 次 batch=1 在每张图上等价（允许 TRT/FP8 微小数值误差）。

```text
【原始 — 2 视角】
  img_base  → get_image_features([1,3,224,224]) → [1,256,D]   ← 第 1 次
  img_wrist → get_image_features([1,3,224,224]) → [1,256,D]   ← 第 2 次
  cat → prefix_embs

【优化 — batch 一次】
  stacked = cat([img_base, img_wrist], dim=0)     → [2,3,224,224]
  feats   = get_image_features(stacked)          → [2,256,D]   ← 只 1 次
  split + cat language → prefix_embs              （语义相同）
```

#### 2.1.3 实现链路（三层配合）

**① ONNX 导出 — 放开 batch 动态轴**（`models/pi05/vit.py`）

```python
dynamic_axes={
    "pixel_values": {0: "batch_size"},
    "image_features": {0: "batch_size"},
}
```

子图：`vision_tower → multi_modal_projector → /sqrt(hidden_size)`，允许 batch=N。

**② TRT 编译 — 按视角数设 profile**（`config/build_configs/vit_build_cfg.py`）

```python
_NUM_VIEWS = 3   # 部署视角数不同则改此值并重编 engine
min_shapes:  pixel_values (1, 3, 224, 224)
opt/max:     pixel_values (_NUM_VIEWS, 3, 224, 224)
```

产物如 `vit_fp8_batch.engine`（区别于旧 batch=1 静态引擎）。实际视角数 > `_NUM_VIEWS` 会 shape 报错。

**③ 运行时 — batched embed_prefix**（`pi05_trt_engine_setup.py`）

`vit_batch_views: true` 时装 `embed_prefix_vit_batched`，替换原版循环：

```python
stacked = torch.cat(images, dim=0)                    # [N*B, 3, H, W]
feats = paligemma_model.get_image_features(stacked)   # 1 次 TRT
for i, img_mask in enumerate(img_masks):
    img_emb = feats[i * bview : (i + 1) * bview]      # 按视角切回
    ...
# language embed + cat（与原版 embed_prefix 后半一致）
```

`get_image_features` 已 hook 为 TRT engine；`trt_vit_scale_fix: true` 时在返回后补乘 `×sqrt(hidden_size)` 与 PyTorch 对齐。

Profiling：`trt.vit.get_image_features` 从每帧 **N 次** 变为 batched 路径内 **1 次** vision 调用。

#### 2.1.4 加速来源

| 维度 | 逐视角 N 次 | 多视角 batch 1 次 |
|------|------------|------------------|
| TRT/ViT 调用 | N | **1** |
| Kernel launch | N ×（SigLIP 27 层 + projector） | **1 ×** |
| GPU 利用率 | batch=1，SM 偏空 | batch=N，GEMM/卷积更满 |
| CUDA Graph | 可能 N 次 replay | **1 次**（固定 N） |
| Python 调度 | N 次 Python→TRT | **1 次** |

2～3 视角场景下 ViT 段延迟常可接近按视角数比例下降（需 profile 验证）。

#### 2.1.5 前置条件与互斥

| 要求 | 说明 |
|------|------|
| `vit_engine` 支持动态 batch | 须 `vit_fp8_batch.engine`，不能用 batch=1 静态引擎 |
| 视角数 ≤ `_NUM_VIEWS` | 编译 profile 上限 |
| **互斥** `embed_prefix_engine` | 整图 embed_prefix TRT 已含 vision |
| **互斥** `use_flashrt_siglip_embed_prefix` | FlashRT SigLIP 走另一 batched 路径 |

附带：`vit_batch_views` 开启时会尝试 `maybe_install_fp8_lang_embedding`，prefix 阶段 language lookup 可走 FP8 sidecar。

### 2.2 LLM：KV-only 输出 + FP8

**原始**：prefill 走 HF `forward(..., use_cache=True)`，并设 `_attn_implementation = "eager"`（纯 PyTorch 注意力，无 FlashAttention）。

**本配置**：`llm_engine: llm_kv_fp8.engine`

1. **FP8 量化**的 PaliGemma language model engine
2. **KV-only**：engine 只输出 denoise 需要的 `past_key_values`，裁掉 `last_hidden_state` 的输出绑定与 device 拷贝

### 2.3 TRT CUDA Graph

```yaml
trt_cuda_graph: true
trt_cuda_graph_warmup: 3
```

对 ViT/LLM 这类固定 shape、重复调用的 engine 做 CUDA Graph capture/replay，消除每帧 Python 调度 + kernel launch 开销。原始 PyTorch 流程每次都是 eager dispatch，无图化。

---

## 三、denoise：FlashRT decoder 的多重优化

相对原始流程优化最密集的部分。原始 `denoise_step` 每步执行：

- `embed_suffix`（action_in_proj + time_mlp → adarms_cond）
- 重建 prefix/suffix 2D/4D attention mask
- `paligemma_with_expert.forward(..., suffix_embs)`（18 层 Gemma Expert）
- `action_out_proj`

### 3.1 整 10 步循环融合（vs 逐步 Python dispatch）

| | 原始 PyTorch | FlashRT |
|--|-------------|---------|
| 调度 | Python `while` 循环，10 次 `denoise_step` | `decoder_forward` 内 C++/CUDA 一次跑完 |
| 每步开销 | 重建 mask、重算 time_emb、HF 18 层 forward | 预分配 buffer + 纯指针 kernel 序列 |
| Euler 积分 | `x_t = x_t + dt * v_t`（Python） | `-1/steps` 烘进 `action_out_proj` 权重 |

实现：`src/model_optimizer/infer/native/flashrt_decoder/pipeline.py::decoder_forward`

### 3.2 算子融合（Kernel Fusion）

原始每层把 RMSNorm、AdaRMS 调制、量化、RoPE、残差、GeGLU 拆成独立 PyTorch op。FlashRT 按「带宽型小块 + 算力型大块」交替融合，**每层从 ~22 kernel 降到 ~13 kernel**：

| 融合 kernel | 替代的原始 op 序列 |
|------------|------------------|
| `fused_adarms_fp8_static` | AdaRMSNorm + amax + quantize + descale |
| `gate_res_adarms_fp8_static` | gate×residual + AdaRMSNorm + quantize |
| `gate_geglu_merged_fp8` | SiLU(gate) × up + quantize |
| `qkv_split_rope_kvcache` | QKV split + RoPE + 写 KV cache |

详见：`docs/optimizer/flashrt/fusion_design.md`

#### 3.2.1 GeGLU 融合（`gate_geglu_merged_fp8_fp16`）

**原始 PyTorch**

Gemma Expert 的 FFN 是标准 GeGLU（`modeling_gemma.py::GemmaMLP`）：

```python
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

denoise 每层、每步的 FFN 段等价于：

```text
x_norm  [S, D]                    # post-attn AdaRMSNorm 输出
  ├─ gate_proj(x_norm)  → gate_h  [S, H]
  ├─ up_proj(x_norm)    → up_h    [S, H]
  ├─ gelu(gate_h)       → g       [S, H]     # act_fn = gelu_pytorch_tanh
  ├─ g * up_h           → hid     [S, H]
  └─ down_proj(hid)     → out     [S, D]
```

PyTorch 里至少 **4～5 个独立 kernel**（2 次 GEMM + gelu + mul + down GEMM），每步都写回 DRAM。denoise 共 18 层 × 10 步 = **180 次**，小 op 的 launch 开销被放大。

**FlashRT 融合路径**

FFN 拆成 **C5（合并 GEMM）+ C6（GeGLU 融合核）+ Down GEMM** 三段（`pipeline.py`）：

| 阶段 | FlashRT | 原始 PyTorch |
|------|---------|-------------|
| C5 | 一次 `fp8_gemm_descale`：`xn × gw → fg[S, 2H]` | 两次 Linear：`gate_proj` + `up_proj` |
| C6 融合核 | `gate_geglu_merged`：读 `fg`，算 GELU×mul，写 `hid_fp8` | `gelu(gate) * up` |
| Down | `fp8_gemm_descale`：`hid_fp8 × dw → fg` | `down_proj` |

**Gate+Up 合并 GEMM（C5）**：权重 repack 时把 `gate_proj` 和 `up_proj` 在输出维 cat 成 `gw [D, 2H]`，一次 GEMM 替代 2 次 Linear + 2 次 `[S,H]` 写回。

**GeGLU 激活核（C6）**：`fg[S, 2H]` 前半是 gate、后半是 up。融合核在一个 kernel 内完成 `gelu(gate) * up`；FP8 路径下同时按 `act_scale_down` 量化，直接产出 Down GEMM 的 FP8 输入，省掉独立 `quantize_fp8`。

**Down GEMM 仍独立**：大矩阵乘属于算力型 op，用 `fp8_gemm_descale` 在 epilogue 做 descale，不与 elementwise 硬融。

**收益**：每层 FFN launch 从 ~5 降到 ~3；gate/up 不再分别落盘；GELU×mul 用 `half2` 读、`uint32` 打包写 FP8（FlashRT 文档约 2.3× 局部加速）。

**注意：GeGLU 的 gate ≠ AdaRMS 的 gate**

- **GeGLU gate**：`gate_proj` 输出，经 GELU 后与 up 相乘 → C6 `gate_geglu_merged`
- **AdaRMS gate**：`input_layernorm.dense(cond)` 第三段，用于 gated residual `x + y * gate` → C1 写入 `bufs['gate']`，C4→5 消费

**本配置（FP16）**：`native_flashrt_use_fp8: false` 时走 `gate_geglu_merged_fp16`，融合结构不变，只是输出 `hid` 为 fp16、不做 FP8 量化。

#### 3.2.2 QKV-RoPE-KV 融合（`qkv_split_rope_kvcache_fp16`）

**原始 PyTorch**

denoise 每层 attention 前（`gemma_pytorch.py`，Expert 分支 `i=1`）：

```text
x_norm [S,D]
  ├─ q_proj / k_proj / v_proj  → Q, K, V
  ├─ reshape / transpose       → [B,NH,S,HD] 等 layout
  ├─ rotary_emb(position_ids)  → cos/sin
  ├─ apply_rotary_pos_emb(Q,K)
  ├─ 写 K/V 到 KV cache（suffix 段）
  └─ attention(Q, Kc/Vc_full)  → 读 prefix+suffix 全长 KV
```

PyTorch 常见 **6～8 个 kernel**。denoise cross-attention 特点：

- **Q** 来自 suffix（action tokens，`S=10`）
- **K/V cache** = prefix（TRT LLM 的 `enc_seq` 段）+ suffix（本层刚写的 `Sa` 段）
- RoPE 位置从**全局 `enc_seq`** 起算，不是从 0 重新编号

**FlashRT 融合路径**

C2 + **C2b 一个融合核**完成 split + RoPE + 写 KV（`pipeline.py`）：

```text
C2:  fp8_gemm_descale(xn, qw) → qkv[S, 2560]     # 合并 QKV GEMM
C2b: qkv_split_rope_kvcache(qkv, rope, attn_out, Kc, Vc, kv_offset)
C3:  attention_qkv_fp16 / FMHA                      # 算力型，独立
```

**C2 合并 QKV GEMM**：权重 repack 把 q/k/v 合成 `qw [D, 2560]`（2560 = 8×256 + 256 + 256）。Q/K 权重做 **pair-interleave**（`interleave_qk`），输出 layout 直接适配 RoPE kernel。

**C2b 融合核内部逻辑**（概念上）：

```text
1. Split qkv → Q[S, Q_dim], K[S, HD], V[S, HD]（按 interleaved 布局解析）
2. 对 Q、K 施 RoPE：读预计算 rope[s, pair*2/2+1] 作为 cos/sin，位置 = enc_seq + s
3. 写 KV cache：Kc[l, enc_seq+s, :] = K_rope；Vc[l, enc_seq+s, :] = V
4. 输出 Q 到 attn_out，layout 对齐 C3 attention
```

`kv_offset = l * total_keys * HD + enc_seq * HD` 指向第 l 层 suffix 写入起点，与 RoPE 表切片 `[enc_seq : enc_seq+Sa)`（`build_dec_rope`）必须一致。详见 `docs/optimizer/flashrt/dec_rope.md`。

**RoPE 预计算**：`setup_prompt` 时 `build_dec_rope(enc_seq, Sa)` 一次性算好 `[Sa, HD]` 交错 cos/sin 表；C2b 只查表 + 旋转，不算三角函数。

**Prefix KV 对齐**：TRT LLM 的 prefix K 写入前需 **pair-interleave**（`fill_prefix_kv_from_trt`），与 suffix 在 C2b 写入的 layout 一致，attention 读完整 `Kc[l, 0:total_keys, :]` 才正确。

**收益**：split + transpose + RoPE + 写 cache 从 4～5 个 kernel → 1 个；GEMM 输出 layout、RoPE、KV 写入、attention 读入统一设计，省显式 transpose。FlashRT 文档：Norm+RoPE+KV 相关 kgen Myelin ~11.9ms → 融合后 ~1.8ms。

**融合边界**：Attention 主体（Q@K^T、softmax、attn@V）属于算力型，刻意独立为 C3，遵循「带宽型融、算力型独立」原则（`fusion_design.md` §3.1）。

#### 3.2.3 单层内融合位置关系

```text
每层 l（单扩散步 s）:

  C1  fused_adarms          → xn_fp8 + gate(AdaRMS)
  C2  QKV GEMM              → qkv[S, 2560]
  C2b qkv_split_rope_kvcache → Q→attn_out, K/V→Kc/Vc[suffix]   ★ QKV-RoPE-KV
  C3  attention
  C4  O proj
  C4→5 gate_res_adarms      → xn_fp8（FFN 输入）
  C5  Gate+Up GEMM          → fg[S, 2H]
  C6  gate_geglu_merged     → hid/hid_fp8                         ★ GeGLU
  C6  Down GEMM             → fg
  C7  gate_res              → 残差进 x
```

#### 3.2.4 FP8 标定槽位（融合边界）

开启 `native_flashrt_use_fp8: true` 时，标定必须按**融合后的张量语义**测 amax（`flashrt_denoise_fp8.md`）：

| 槽位 `l*4+k` | 融合边界 | 标定对象 |
|--------------|---------|---------|
| k=0 (qkv) | C1 `fused_adarms` 输出 | 进入 QKV GEMM 的激活 |
| k=3 (down) | C6 `gate_geglu_merged` 输出 | 进入 Down GEMM 的激活 |

### 3.3 权重 repack / layout 优化

`weights.py` 在加载时一次性把权重重排成 kernel 友好布局，运行时零开销：

- **Fused QKV**：q/k/v 三个投影 cat 成一个 `[2560, D]` GEMM（原始 3 个独立 Linear）
- **Fused Gate+Up**：gate/up cat 成一个 GEMM
- **interleave_qk**：Q/K 输出维预先 pair-interleave，省掉运行时 RoPE layout 转换
- 权重预先转置（`.t()`）成 GEMM 期望布局
- FP8 路径：`quant_fp8` 离线 per-tensor E4M3 量化 + `w_scales`

### 3.4 预计算（Precompute）

**原始**：每步在 `embed_suffix` 里现算 `time_emb` 和 `time_mlp` → `adarms_cond`。

**FlashRT**：`setup_prompt(enc_seq)` 阶段预计算：

- **10 步 × 18 层** AdaRMS 调制 `sa/sf/fs`（`precompute_adarms_styles`）
- **RoPE 表**（`build_dec_rope`）

denoise 热路径只读指针，不算 time_mlp / RoPE。

### 3.5 Buffer 预分配 / 纯指针接口（零 malloc）

**原始**：每步 `torch.cat`、新建 mask、`x_t = x_t + dt * v_t` 分配新张量。

**FlashRT**：`DecoderBuffers` 在 `setup_prompt` 时一次性预分配整个 10 步 × 18 层循环所需的全部中间 buffer，整个 session 复用；kernel 只接收 `data_ptr` 在原地读写。

### 3.6 本配置的精度选择：FP16 路径

```yaml
native_flashrt_use_fp8: false   # fp16 原生精度：不量化、无需标定
native_flashrt_calibrate: false
```

denoise 享受**算子融合 + 整循环 + 预计算 + 零拷贝**的全部收益，但**不含 FP8 量化加速**（GEMM 仍为 fp16）。  
若需更低延迟，改用 `tensorrt_flashrt_denoise.yaml`（`use_fp8: true` + 激活标定），可额外获得 FP8 GEMM 收益（约 -8ms 量级，Thor 参考）。

---

## 四、KV 传递与 Layout 适配

混合路径中 prefix KV 从 TRT LLM 传到 FlashRT decoder：

```text
TRT/PyTorch LLM → DynamicCache
  → _stack_past_key_values()
  → valid_prefix trim / contiguous
  → optional even padding（enc_seq 为奇数时）
  → fill_prefix_kv_from_trt()（K pair-interleave + copy_ 到 Kc/Vc）
  → FlashRT decoder_forward
```

实现：`pi05_executor.py::sample_actions_flashrt`、`flashrt_decoder/driver.py::fill_prefix_kv_from_trt`

---

## 五、显存与权重管理

显存按四类来源分析：权重、激活值、KV cache、TRT engine。

### 5.1 权重显存（已较优）

```yaml
native_enable_expert: false           # expert 权重已 repack 到 FlashRT，不再走 PyTorch
trt_release_pytorch_weights: true     # 释放已被 TRT/FlashRT 接管的 PyTorch 权重
```

加载后按「谁接管谁释放」裁剪 PyTorch 权重（`pi05_executor.py::_release_pytorch_model`）：

- `vit_engine` 设置 → `del paligemma_model.vision_tower`
- `llm_engine` 设置 → 释放 `language_model` layers，保留 `embed_tokens`（语言 embedding 仍需），替换为 TRT forward stub
- denoise FlashRT repack 完 → `del ge.model` / `del ge.lm_head` + `empty_cache()`（`flashrt_decoder` 侧 `_maybe_release_gemma_expert_after_flashrt`）

稳态：ViT/LLM 权重只在 TRT engine（FP8），expert 只在 FlashRT repack buffer，PyTorch 仅留 `embed_tokens`。无大冗余。

**优化空间**

| 点 | 现状 | 优化 |
|----|------|------|
| 加载期峰值 | 释放在 engine/repack **之后**，存在「完整 bf16 权重 + TRT engine + repack 产物」三者共存的瞬时峰值 | 逐层 repack 后立即释放该层 PyTorch 权重，削平 peak（Thor 显存紧张时优先级高） |
| repack 临时副本 | `weights.py` 每层 `.float()` → `cat` → `.t().contiguous()` 产生 fp32 临时张量 | 逐层处理后即释放，减少同时存活的 fp32 副本 |
| embed_tokens | 全词表 fp16/bf16 常驻（PaliGemma 词表大） | FP8 embedding（参考 TensorRT-Edge-LLM `--fp8-embedding`，约减半） |

### 5.2 激活值显存（denoise，已接近最优）

`DecoderBuffers` 在 `setup_prompt` 一次性预分配、整个 session 复用，热路径零 malloc（`driver.py`）：

- `x / xn / gate / qkv / logits / attn_out / hid / fg` + FP8 buffer
- 因 `S = Sa = 10`（action token 数）极小，全部为 **KB 级**，绝对占用可忽略
- 已做原地复用：`attn_out` 既作 Q 又接输出；`fg` 在 O-proj / gate-up / down 输出间轮用

**优化空间（收益小，主要是干净度）**

- 校准 scratch（`calib_buf / d_scale / hidden_scratch / fp8_scratch`）无论是否标定都分配 → 正式推理（`calibrate: false`）时可惰性分配或标定后释放
- FP16 路径下仍分配 `xn_fp8 / hid_fp8 / ctx_fp8` → 可按 `use_fp8` 条件分配

> 结论：denoise 激活因 S=10 无压缩价值，大头在 KV cache 与 TRT engine。

### 5.3 KV cache（最大优化空间）

FlashRT 侧 KV slab（`driver.py`，单 KV head / GQA）：

```text
Kc / Vc : [layers, total_keys, HD] fp16
        = [18, ~828, 256] → K+V 合计约 14.6 MB
```

prefix KV 从 TRT LLM 到 FlashRT 每 chunk 的搬运链路：

```text
TRT 输出 → DynamicCache → _stack_past_key_values(torch.cat)
  → slice + .contiguous()
  → 奇数 enc_seq 时 even-pad torch.cat
  → fill_prefix_kv_from_trt(pair-interleave reshape/permute + copy_)
  → Kc/Vc prefix 区
```

即每帧存在 **3~4 次中间拷贝**（靠 caching allocator 复用，但仍有带宽与 allocator churn）。

**优化空间（高收益）**

| 点 | 现状 | 优化 | 收益 |
|----|------|------|------|
| KV 搬运 | 每帧 stack→contiguous→interleave→copy | TRT output binding 直接写 Kc/Vc prefix 区，或图内完成 pair-interleave | 带宽 + churn ↓ |
| KV 精度 | fp16（~14.6MB） | FP8 KV cache（减半 + attn 读带宽↓） | 显存 + 延迟（参考 Edge-LLM FP8 KV，Blackwell context attn 快 9~17%） |
| even-pad | 奇数 enc_seq 每帧 `cat` | prompt/engine 层固定偶数 enc_seq | 每帧一次拷贝↓ |
| valid_prefix | 每帧 `[:, valid_prefix, :].contiguous()` bool indexing | 固定 padding，使物理 KV 直接等于有效 token | 每帧拷贝↓ |

### 5.4 TRT engine 显存

CUDA Graph 模式每个 engine 缓存 static input/output（固定 shape 仅 1 个 key），但每帧 replay 后 **clone 输出**（`trt_torch.py`，防下次 replay 覆盖）。

**优化空间**

| 点 | 现状 | 优化 |
|----|------|------|
| 输出 clone | 每帧 `tensor.clone()` 分配新输出（LLM KV 输出尤其大） | double-buffer 两套固定输出轮替（与 5.3 KV 零拷贝衔接） |
| context memory | ViT/LLM engine 各自持有 execution context scratch | 串行执行 → 共享一块 `max(engine 所需)` context memory（参考 Edge-LLM shared context memory） |
| KV-only | 本配置 `llm_kv_fp8.engine` 已只绑定/输出 KV，省 `last_hidden_state` 输出 buffer 与拷贝 | 已做 |

### 5.5 小结

| 区域 | 优化点 | 收益 | 优先级 |
|------|--------|------|--------|
| KV cache 搬运 | TRT 输出零拷贝到 Kc/Vc | 带宽 + churn ↓ | 高 |
| KV cache 精度 | FP8 KV | 显存 + 延迟 | 中高 |
| even-pad / valid_prefix | 固定偶数 enc_seq + 固定 padding | 每帧拷贝↓ | 中（易做） |
| TRT 输出 | double-buffer | 每帧分配↓ | 中 |
| engine context | 共享 context memory | 常驻显存↓ | 中 |
| 加载峰值 | 逐层 repack 即释放 + 削减 fp32 临时 | peak↓（防 OOM） | 中（Thor 紧张时高） |
| embed_tokens | FP8 embedding | 常驻显存↓ | 低中 |
| 校准 / FP8 scratch | 惰性 / 条件分配 | 少量 | 低 |

> 总体判断：激活值（denoise，S=10）已无优化必要；最值得做的是 **KV cache 链路零拷贝 + FP8 KV**（既省显存又降延迟），其次是 **TRT engine 输出 double-buffer 与共享 context memory**，以及 **加载期峰值** 削减（Thor 显存紧张时上升为高优先级）。

---

## 六、量化原理（FP8 静态标定 + NVFP4）

> ViT/LLM engine 与 denoise FP8 路径（`tensorrt_flashrt_denoise.yaml`，`use_fp8: true`）共用同一套 FP8 量化思想。  
> 实现：`flashrt_decoder/weights.py`（权重）、`pipeline.py`（前向 + 标定）、`backend.py`（多样本累计）。  
> NVFP4（Blackwell/Thor）见 §6.8。

### 6.1 FP8 格式

| 格式 | 位分配 | 范围 | 用途 |
|------|--------|------|------|
| **E4M3** | 1 符号 + 4 指数 + 3 尾数 | ±448 | 权重 + 激活（前向推理，本仓库用此） |
| E5M2 | 1 符号 + 5 指数 + 2 尾数 | ±57344 | 梯度（训练） |

E4M3 满量程 **±448** 是后续 scale 的分母来源。FP8 仍是浮点（带指数位），对跨数量级的激活比 INT8 更鲁棒。

### 6.2 scale 与 zero_point

量化映射：`x_q = round(x / scale) + zero_point`，反量化 `x ≈ scale * (x_q - zero_point)`。

| 方式 | zero_point | 适用 |
|------|-----------|------|
| 非对称 | ≠ 0 | INT8，分布偏一侧（如 ReLU 后全正） |
| **对称** | **= 0** | **FP8**，权重/Norm 后激活近似零对称 |

**FP8 用对称量化，zero_point 恒为 0**：FP8 浮点零点天然精确、无需偏移；Norm 后激活与权重近似零对称。所以只有一个 scale，GEMM 里没有 INT8 非对称的 zero_point 交叉项。

```python
# weights.py::quant_fp8 —— FP8 E4M3 per-tensor 对称量化
a = w.float().abs().max().item()      # amax
s = max(a / 448.0, 1e-12)             # scale = amax / 448（满量程对齐）
return (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s
```

### 6.3 权重量化（离线 / 静态 / per-tensor）

加载时一次性量化，整个推理不变。Pi0.5 decoder 每层 4 个 GEMM（qkv/o/gate-up/down）→ 4 个权重 scale，存进 `ae_w_scales [layers*4]`。per-tensor 是 E4M3 ONNX 导出的约束（per-channel E4M3 不被支持）。

### 6.4 激活量化（在线标定 scale / 静态应用）

权重固定、scale 一算即得；激活随输入变，需**标定**确定 scale。

| 策略 | scale 来源 | 开销 |
|------|-----------|------|
| 动态量化 | 每次前向实时算 amax | 每 GEMM 多 amax reduce + device sync（慢） |
| **静态量化（FlashRT）** | 离线标定固化为常量 | 推理时 0 额外 kernel |

每层 4 个激活量化点与权重 scale 一一对应：`l*4+k`，k=0 qkv / 1 o_proj / 2 gate_up / 3 down。

### 6.5 实际计算：quantize → FP8 GEMM → descale

对 `Y = X @ W`，两者对称量化后：

```
Y = X @ W ≈ (s_act·X_q) @ (s_w·W_q) = (s_act·s_w) · (X_q @ W_q)
                                        └标量┘   └ FP8 Tensor Core GEMM ┘
```

三步：① 用 `s_act` 把激活量化成 FP8；② `X_q @ W_q` 在 Tensor Core 跑（**FP32 累加**）；③ 乘标量 `s_act·s_w` descale 回 fp16。descale 是 GEMM 的 **epilogue**，不是单独 kernel。

```python
# pipeline.py：C1 产出 FP8 激活，C2 做 FP8 GEMM+descale 直接输出 fp16
fvk.fused_adarms_fp8_static_fp16(x, sa_ptr, xn_fp8, gate, S, D, act_scale_qkv, stream)
fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D, act_scale_qkv, w_scale_qkv, stream)
```

**vs 动态量化**：动态每个 GEMM 要 `amax→scale→quant→GEMM→dequant`（4 kernel + 1 sync）；静态只剩 `GEMM(descale=常量)`，全 pipeline 省约 630 次 quant/dequant kernel。

### 6.6 两层 max-merge（激活标定鲁棒性增强）

**问题**：静态激活 scale = `amax/448` 固化后不变。若推理时真实激活 > 标定 `amax`，则 `x/s_act > 448` 被 **clamp 饱和**，大值削平 → 掉点。所以标定要让固化的 `amax` 尽量覆盖推理峰值。

FP8 E4M3 范围窄（±448），对 amax 极敏感：scale 偏小→饱和掉点；scale 偏大（被 outlier 拉高）→小值分辨率下降。max-merge 在两个维度取 max，得到「上包络」：

**第一层 — 跨扩散步 max（单样本内）**

denoise 10 步里同一量化点激活幅度不同。
- 原版 FlashRT：只留最后一步（step-9）amax。
- 改进：每步 amax 落盘 `per_step_scales_ptr`，对 step 轴取 max（`pipeline.py::decoder_forward_calibrate` + `driver.py::calibrate` 的 `per_step.amax(dim=0)`）。
- 注意：标定循环内 FP8 前向仍用当步 scale（与原版一致），只有**导出的 scale** 取跨步 max。

**第二层 — 跨样本 max（多 observation）**

`backend.accumulate_calibration` 每个样本（KV/noise 各异）算完跨步 max 后，再与累计值 `torch.maximum`；`reset_act_scales` 清零起点，累计 `native_flashrt_calib_samples`（默认 8）个样本后 `save_act_scales` 固化。

```
最终 act_scale[l*4+k] = max over (N 样本) of [ max over (10 步) of amax ]
                        └──── 第二层 ────┘    └───── 第一层 ─────┘
```

**与正常做法的区别**

| | 原版 | 两层 max-merge |
|--|------|---------------|
| 扩散步 | 只取最后一步 | 10 步取 max |
| 样本 | 单样本/单 noise | N=8 取 max |
| scale 倾向 | 可能偏小 | 保守上包络 |
| 风险 | 易饱和掉点 | 饱和概率大降 |

**为何增强鲁棒性**：在受控有限集合（10 步 × 8 样本 = 80 观测）上取 max，既不像单点那样偏小（→饱和），也不像无界 single-max 那样被极端 outlier 拉爆（样本有限、denoise 激活分布相对稳定），在「饱和」与「分辨率」间取更稳折中。**FP8 掉点时增大 `native_flashrt_calib_samples`** 即多看样本、上包络更可靠。

### 6.7 关键结论

| 概念 | 本仓库 FP8 做法 |
|------|----------------|
| 格式 | E4M3，±448 |
| zero_point | **恒为 0**（对称量化） |
| scale | `amax / 448`，权重 + 激活各一组 |
| 权重 | 离线、per-tensor、静态 |
| 激活 | 离线标定（两层 max-merge），推理时常量 |
| GEMM | `Y = (X_q @ W_q)·(s_act·s_w)`，FP32 累加，epilogue descale |
| 加速 | 静态 scale 省 amax/quant/dequant + sync；Tensor Core FP8 算力翻倍 |

### 6.8 NVFP4（4-bit 块量化，Blackwell / Thor）

> 配置：`config/quant/llm_quant_nvfp4_*.py`、`denoise_quant_nvfp4_fp8_mix_cfg.py` 等  
> 量化 kernel：`third_party/FlashRT/csrc/quantize/bf16_weight_to_nvfp4_swizzled.cuh`

NVFP4 是 Blackwell 的 **4-bit 浮点块量化（W4A4）**，核心是「**4-bit 数据 + 两级 scale**」。**同样是对称量化，zero_point = 0**。

#### 6.8.1 三件套：数据 + 两级 scale

最终用三个量近似原始权重：

```
w  ≈  global_scale  ×  SF_block  ×  x_e2m1
      └ per-tensor ┘   └ per-16 ┘   └ 4bit ┘
        FP32 标量      FP8 E4M3      E2M1
```

| 级别 | 粒度 | 存储 | 作用 |
|------|------|------|------|
| global_scale | per-tensor（整张量 1 个） | FP32 | 把 block scale 压进 FP8 范围 |
| SF（scale factor） | **每 16 个元素 1 个** | FP8 E4M3（≤448） | 每小块独立适配幅度 |
| 数据 | per-element | E2M1 4bit | 实际值 |

- **E2M1**（1 符号 + 2 指数 + 1 尾数）只能取 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`，最大 ±6。
- **为何 per-16-block**：矩阵里少数 outlier 会把 per-tensor 单 scale 拉大，导致小值被量化成 0。NVFP4 每 16 元素一块、各自 scale，outlier 只影响所在块，小值不被拖死——**用细粒度 scale 换回低位宽损失的精度**。
- **为何两级**：block scale 用 FP8（1 字节）存以省空间/便于硬件读，但 FP8 上限 448，某些块理想 scale 会超；再提一个全局 FP32 `global_scale` 把所有 block scale 压进 FP8 范围。

#### 6.8.2 量化过程（两遍法 + 数值示例）

设某权重张量 `global_amax = 6.0`。

**Pass 1（定全局 scale）**

```
global_scale = global_amax / 2688 = 6.0 / 2688 ≈ 0.002232
```

`2688 = 448（FP8 E4M3 max）× 6（E2M1 max）`，推导：含最大值的块理想有效缩放 = `global_amax/6`，要让它对应的 SF 正好 = FP8 上限 448 → `global_amax/6 = global_scale × 448` → `global_scale = global_amax/2688`。这样最大块 SF=448 不溢出，其余块 SF<448。

**Pass 2（逐 16 元素块）**

块 A（含全局最大，block_amax=6.0）：
```
SF_A = block_amax / (6 × global_scale) = 6.0 / 0.013392 = 448   ← FP8 满档 ✓
有效缩放 = global_scale × SF_A = 0.002232 × 448 = 1.0
w=6.0 → round(6.0/1.0)=6 ；w=3.0 → 3 ；w=1.5 → 1.5
```

块 B（全小值，block_amax=0.75）：
```
SF_B = 0.75 / 0.013392 ≈ 56.0   ← FP8 存得下 ✓
有效缩放 = 0.002232 × 56 = 0.125 （= block_amax/6 = 0.75/6）
w=0.75 → round(0.75/0.125)=6  ← 小值块也用满 E2M1 范围 ✓
w=0.25 → round(0.25/0.125)=2  dequant 2×0.125=0.25 ✓
```

> 对比：若用 FP8 那种全局单 scale（=1.0），块 B 的 `w=0.25 → round(0.25/1.0)=0` 直接丢失；NVFP4 因块 B 有自己的小有效缩放（0.125）仍精确。这就是细粒度 scale 的价值。
>
> 本质：`x_e2m1 = round(w/(global_scale×SF)) = round(w×6/block_amax)`，即把每块最大值映射到 E2M1 的 6，让每块都用满 4-bit。

**收尾**：2 个 e2m1 拼进 1 字节；SF 按硬件 swizzle 布局写出（便于 Tensor Core 读取）。

#### 6.8.3 推理计算

FP4 Tensor Core 做 `Y = X @ W` 时按块反量化：`w_real ≈ global_scale × SF_block × x_e2m1`，激活同为 block-scaled NVFP4（W4A4），FP32 累加。

#### 6.8.4 NVFP4 vs FP8

| 维度 | FP8 (E4M3) | NVFP4 |
|------|-----------|-------|
| 数据位宽 | 8 bit | **4 bit**（E2M1） |
| 数据最大值 | ±448 | ±6 |
| scale 粒度 | per-tensor 单 scale | **两级：per-tensor FP32 + per-16 FP8** |
| zero_point | 0（对称） | 0（对称） |
| 量化对象 | W8A8 | **W4A4** |
| 算力（Blackwell） | ~2× BF16 | **~4× BF16（~2× FP8）** |
| 显存/带宽 | 1/2 BF16 | **1/4 BF16** |
| 硬件 | SM89+ | **SM100+/Blackwell（Thor sm_110）** |
| 精度 | per-tensor 较稳 | 位少，靠 block-16 细 scale 补偿；敏感层常需 AWQ |
| attention | 可 FP8 attention | **FP4 不跑 attention**，q/k/v BMM 退 FP8；KV cache FP8 |

#### 6.8.5 本仓库 NVFP4 策略

- **延时优先**（`llm_quant_nvfp4_latency_cfg.py`）：全 Linear 走 NVFP4，最大化 FP4 GEMM 覆盖。
- **mix**（`*nvfp4_fp8_mix_cfg.py`）：敏感层（如 11–17 层）降 FP8 平衡精度。
- **AWQ 兜底**：per-channel `pre_quant_scale` 折进权重、运行期≈0 开销，优于直接降 FP8。
- attention 用 FP8、KV cache 用 FP8（FP4 不适合 attention，KV 用 FP8 压带宽）。
- 性能参考（FlashRT，Pi0.5 Thor）：NVFP4 ≈ **39.78ms / 25Hz** vs FP8 **44ms / 23Hz**。

---

## 七、注意力实现对比

| | 原始 PyTorch | 本配置 |
|--|------------|--------|
| LLM prefill attention | HF `eager`（matmul + softmax） | TRT engine 内部优化 attention kernel |
| denoise attention | HF eager，每步重建 4D additive mask | FlashRT `attention_qkv_fp16` / 可选 CUTLASS FMHA |

本配置 `native_flashrt_fmha_so: ""` 未启用可选 FMHA。`kernelSrc/fmha_d256_cutedsl` 为 d=256 CuTe DSL FMHA（Thor sm_110），可进一步接入 prefill/denoise。

---

## 八、优化手段汇总表

| 维度 | 原始 PyTorch pi05 | 本配置优化 | 收益类型 |
|------|------------------|-----------|---------|
| **后端异构** | 单一 PyTorch/HF | ViT=TRT, LLM=TRT, denoise=FlashRT | 各阶段最优 |
| **TRT engine** | 无 | ViT + LLM 编译期图优化 | 算子融合 / tactic |
| **量化** | bf16/fp32 | ViT FP8、LLM FP8（denoise 本配置 fp16） | 算力 / 带宽 |
| **多视角 batch** | 逐视角 N 次前向 | 一次 batched ViT | launch ↓ |
| **KV-only LLM** | 输出含 hidden_state | engine 只出 KV | 拷贝 / 绑定 ↓ |
| **CUDA Graph** | 无 | TRT engine capture/replay | dispatch ↓ |
| **denoise 整循环** | 10 次 Python denoise_step | 1 次 C++ 整循环 | dispatch ↓↓ |
| **算子融合** | 逐 op | AdaRMS/GeGLU/QKV-RoPE-KV 融合，22→13 kernel/层 | launch + DRAM ↓ |
| **权重 repack** | 标准 Linear | fused QKV/GateUp + interleave_qk + 转置 | GEMM 效率 |
| **预计算** | 每步算 time_mlp/RoPE | setup 期预算 sa/sf/fs + RoPE 表 | 重复计算 ↓ |
| **buffer 复用** | 每步新张量 | 预分配 + 纯指针原地写 | malloc ↓ |
| **Euler 积分** | Python 逐步更新 | `-1/steps` 烘进 out_proj 权重 | op ↓ |
| **显存** | 全量权重常驻 | 释放被接管的 PyTorch 权重 | 显存 ↓ |

---

## 九、端到端数据流（混合路径）

```text
Policy.infer(obs)
  → _preprocess_observation
  → embed_prefix（TRT ViT batch + PyTorch language embed + 拼接）
  → prefix_llm（TRT LLM prefill → KV cache）
  → KV 适配（stack → trim → RoPE interleave → copy 到 FlashRT Kc/Vc）
  → FlashRT denoise 整 10 步循环（decoder_forward）
  → action chunk
```

---

## 十、本配置已启用 vs 未启用

### 已启用

- TRT(vit/llm) + 算子融合 + denoise 整循环 + 预计算 + buffer 复用
- ViT 多视角 batching、LLM KV-only engine、TRT CUDA Graph
- 显存优化（释放 PyTorch 权重）

### 未启用（可进一步降低延迟）

| 能力 | 说明 | 参考 |
|------|------|------|
| denoise FP8 | `native_flashrt_use_fp8: true` + 激活标定 | `tensorrt_flashrt_denoise.yaml` |
| FlashRT 整 pipeline CUDA Graph | 10 步 unroll 一次 capture + autotune | FlashRT `record_infer_graph` |
| cuBLASLt autotune | `autotune=3` 缓存最优 tactic | FlashRT frontend |
| NVFP4 decoder | Thor 全栈约 39.78ms vs FP8 44ms | FlashRT `use_fp4` |
| CUTLASS FMHA | `native_flashrt_fmha_so` 指向 `.so` | `build_flashrt_kernels.sh` |
| embed_prefix 整图 TRT | image + language + concat 一段 TRT | `embed_prefix.engine` |
| KV 零拷贝 | TRT 直接输出到 FlashRT Kc/Vc slab | `docs/optimizer/phase2/rm.md`、本文 §5.3 |
| FP8 KV cache | Kc/Vc 由 fp16 降 FP8（减半 + attn 带宽↓） | 本文 §5.3 |
| TRT 输出 double-buffer / 共享 context memory | 去每帧 clone、共享 engine scratch | 本文 §5.4 |

---

## 十一、相关文件

| 文件 | 角色 |
|------|------|
| `config/webui_configs/tensorrt_native_denoise.yaml` | 本总结对应的示例配置 |
| `third_party/openpi/.../pi0_pytorch.py` | 原始 PyTorch 推理流程 |
| `src/model_optimizer/infer/native/pi05_executor.py` | FlashRT hybrid 主循环（`sample_actions_flashrt`） |
| `src/model_optimizer/infer/native/flashrt_decoder/` | denoise 整循环 backend / pipeline / weights |
| `src/model_optimizer/infer/tensorrt/pi05_trt_engine_setup.py` | ViT/LLM TRT hook 安装（含 `embed_prefix_vit_batched`） |
| `src/model_optimizer/models/pi05/vit.py` | ViT ONNX 导出（batch 动态轴） |
| `config/build_configs/vit_build_cfg.py` | ViT TRT 编译 profile（`_NUM_VIEWS`） |
| `config/webui_configs/README.md` | `vit_batch_views` 用法与互斥说明 |
| `docs/optimizer/flashrt/fusion_design.md` | Decoder 融合算子设计 |
| `docs/optimizer/flashrt/dec_rope.md` | suffix RoPE 预计算与 C2b KV 写入 |
| `docs/optimizer/ddup/flashrt_denoise_fp8.md` | denoise FP8 启用与标定 |
| `config/quant/llm_quant_nvfp4_*.py` | LLM NVFP4 量化配置（latency / fp8-mix / awq） |
| `third_party/FlashRT/csrc/quantize/bf16_weight_to_nvfp4_swizzled.cuh` | NVFP4 权重量化 kernel（两遍法 + swizzle） |
| `docs/optimizer/phase2/rm.md` | Prefix / KV 链路 Phase 2 优化方向 |
