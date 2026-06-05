# Pi0.5 推理 Pipeline 分析

本文档总结 OpenPI `PI0Pytorch`（Pi0.5）从观测到 action 的推理数据流，涵盖 Flow Matching 去噪、AdaRMSNorm 时间条件化、attention 掩码、RoPE 位置编码及与 FlashRT 优化的关系。

主要代码入口：

- `third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py` — `sample_actions` / `denoise_step` / `embed_prefix` / `embed_suffix`
- `third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` — RoPE / adaRMS / attention
- `src/model_optimizer/infer/native/flashrt_decoder/precompute.py` — AdaRMS 离线预计算

---

## 1. 总览：两阶段推理

Pi0.5 推理分为 **Prefix 编码（一次）** 和 **Denoise 循环（`num_steps` 次，默认 10）**：

```
observation
    │
    ├─► embed_prefix(image + language) ──► PaliGemma LM ──► past_key_values（KV cache）
    │
    └─► noise ~ N(0,I)  →  x_t
              │
              ▼
        ┌─ denoise loop × num_steps ─────────────────────────┐
        │  denoise_step(x_t, t) → v_t                        │
        │  x_t = x_t + dt * v_t                              │
        │  t  = t + dt                                       │
        └────────────────────────────────────────────────────┘
              │
              ▼
        actions [B, action_horizon, action_dim]
```

| 阶段 | 计算内容 | 频率 |
|------|----------|------|
| Prefix | 图像 SigLIP + 语言 token → PaliGemma language model | 每帧 1 次 |
| Denoise | action expert（Gemma 300M）读 prefix KV，预测速度场 `v_t` | 每帧 `num_steps` 次 |

---

## 2. Flow Matching 与 `dt`

### 2.1 Euler 积分

```python
dt = -1.0 / num_steps          # num_steps=10 → dt=-0.1
x_t = noise                    # t=1，纯噪声
time = 1.0
while time >= -dt / 2:         # t = 1.0, 0.9, …, 0.1，共 10 步
    v_t = denoise_step(..., x_t, time)
    x_t = x_t + dt * v_t       # Euler 更新
    time += dt
```

- **约定**：`t=1` 为噪声，`t=0` 为目标 action（与部分 diffusion 文献相反，见源码注释）
- **`dt` 负号**：时间从 1.0 反向积分到 0.0
- **`dt` 转 tensor**：与 GPU 上 `x_t`、`v_t` 同 device/dtype，避免混算开销

### 2.2 每步语义

模型在时刻 `t` 预测速度场 `v_t`；`x_t + dt·v_t` 沿 ODE 走向更干净的 action。

---

## 3. State 输入：Pi0 vs Pi0.5

| | **Pi0** | **Pi0.5** |
|---|---------|-----------|
| state 位置 | suffix 第 1 个 token（`state_proj` 连续嵌入） | **不在 suffix** |
| state 编码 | 连续向量 → action expert | 离散化（256 bin）→ 写入语言 prompt |
| prompt 格式 | 仅 task 文本 | `Task: {prompt}, State: {bins};\nAction: ` |
| `embed_suffix(state, …)` | 使用 `state` | **`state` 被忽略**（API 兼容保留参数） |

Pi0.5 的 robot state **仍然参与生成**，只是经 **prefix 语言 token** 进入模型，而非 suffix 连续 token。推理链路须保证 `TokenizePrompt(discrete_state_input=True)` 正确执行。

配置见 `Pi0Config`（`pi0_config.py`）：

```python
# pi05=True 时默认 discrete_state_input=True
# state 是离散语言 token 的一部分，而非 suffix 连续输入
# action expert 用 adaRMSNorm 注入 flow matching 时间步
```

---

## 4. 时间条件化：`sincos` → `time_mlp` → `adarms_cond`

### 4.1 正弦位置编码（标量时间 → 向量）

`create_sinusoidal_pos_embedding(time, dimension, min_period=4e-3, max_period=4.0)`：

1. 在 `[min_period, max_period]` 上 log 均匀取 `dimension/2` 个周期 \(T_i\)
2. 角频率 \(\omega_i = 2\pi / T_i\)，相位 \(\phi_i = \omega_i \cdot t\)
3. 输出 `[sin(φ), cos(φ)]` 拼接，维度 = `dimension`（= action expert hidden width）

多频率 sin/cos 使模型在 \(t \in [0,1]\) 的多时间尺度上分辨去噪进度。

### 4.2 Pi0.5：adaRMS 注入

```python
time_emb = sincos(timestep)
time_emb = silu(silu(time_emb @ W_in + b_in) @ W_out + b_out)   # time_mlp
adarms_cond = time_emb                                          # [B, Da]
```

`adarms_cond` 进入每层 `GemmaRMSNorm`，产生 `(scale, shift, gate)` 调制 Norm 与 residual（**不是独立 attention token**）。机制详见 **第 5 章**。

Pi0 则把 time 与 action_emb concat 后过 MLP，无 adaRMS。

### 4.3 离线预计算（`num_steps` 固定时）

推理时 `t ∈ {1.0, 0.9, …, 0.1}` 仅 10 个固定值，`adarms_cond` **只依赖步数，与 `x_t`/观测无关**：

| 层级 | 预计算内容 | 实现 |
|------|-----------|------|
| L1 | `sincos(t)` | 任意后端 |
| L2 | + `time_mlp` → `adarms_cond[10, Da]` | PyTorch 可省每步 MLP |
| L3 | + 各层 `dense(adarms_cond)` → `sa/sf/fs` | FlashRT `precompute_adarms_styles()` |

FlashRT 在 build/setup 时完成 L3，推理 C1 `fused_adarms` 直接读表；TRT 路径有 `AdaRmsModulator` 按 `timestep` 缓存。

**注意**：训练时 `time ~ Beta(1.5, 1)` 连续随机，无法穷举预计算。

---

## 5. AdaRMSNorm 详解

Pi0.5 action expert 使用 **AdaRMSNorm（Adaptive RMS Normalization）** 注入 Flow Matching 时间条件。实现位于 `modeling_gemma.py` 的 `GemmaRMSNorm.forward`；Pi0.5 配置 `use_adarms=[False, True]`，仅 **Gemma Expert（300M）** 启用，`cond_dim = hidden_size`。

### 5.1 核心代码（`dense(cond)` → scale / shift / gate）

```python
# modeling_gemma.py — GemmaRMSNorm.forward（Pi0.5 路径，cond 非空）
normed_inputs = self._norm(x)                    # ① RMSNorm

modulation = self.dense(cond)                    # ② Linear(Da → 3D)
if len(x.shape) == 3:                            # ③ [B, 3D] → [B, 1, 3D]
    modulation = modulation.unsqueeze(1)         #    沿 seq 维广播到 H 个 action token

scale, shift, gate = torch.chunk(modulation, 3, dim=-1)  # ④ 三等分

normed_inputs = normed_inputs * (1 + scale) + shift        # ⑤ 仿射调制
return normed_inputs.to(dtype), gate.to(dtype)             # ⑥ gate 留给 gated residual
```

| 步骤 | 输入形状 | 输出形状 | 含义 |
|------|----------|----------|------|
| ① `_norm(x)` | `[B, H, D]` | `[B, H, D]` | RMSNorm：\(\hat{x} = x / \sqrt{\mathrm{mean}(x^2)+\epsilon}\) |
| ② `dense(cond)` | `cond [B, Da]` | `[B, 3D]` | 时间条件 → 调制向量（`Da = D`） |
| ③ `unsqueeze(1)` | `[B, 3D]` | `[B, 1, 3D]` | 同一 denoise 步内，H 个 token 共享同一组调制 |
| ④ `chunk(3)` | `[B, 1, 3D]` | 各 `[B, 1, D]` | 拆成 scale / shift / gate |
| ⑤ 仿射变换 | — | `[B, H, D]` | \(\hat{x} \odot (1+\mathrm{scale}) + \mathrm{shift}\) |
| ⑥ 返回 gate | — | `[B, 1, D]` | 用于 sublayer 后的 gated residual |

`dense` 权重 **零初始化**（`nn.init.zeros_(self.dense.weight)`）：训练初期 scale≈0、shift≈0、gate≈0，行为接近普通 RMSNorm + 标准 residual，便于从预训练权重平滑过渡。

### 5.2 scale / shift / gate 各自作用

**scale + shift** — 条件化仿射变换（类似 FiLM / DiT 的 adaLN）：

\[
\text{out} = \mathrm{RMSNorm}(x) \odot (1 + \mathrm{scale}) + \mathrm{shift}
\]

- **scale**：按 hidden 通道缩放（乘性）
- **shift**：按 hidden 通道平移（加性）

两者由当前去噪时间 `t` 经 `time_mlp` 得到的 `adarms_cond` 决定，告诉模型「这一步应按什么统计特性处理 hidden states」。

**gate** — 控制 residual 通量，在 sublayer（Attention / FFN）输出与 skip connection 合并时使用：

```python
# modeling_gemma.py — _gated_residual
return x + y * gate    # x=residual, y=sublayer 输出
```

每层 **两次** gated residual：`input_layernorm` 后（Attention 分支）、`post_attention_layernorm` 后（FFN 分支）。最终还有 `model.norm(hidden_states, adarms_cond)` 做一次 AdaRMS（无后续 gate）。

### 5.3 在 denoise 中的调用位置

`adarms_cond` 来源（`pi0_pytorch.py` `embed_suffix`）：

```python
time_emb = silu(silu(sincos(t) @ W_in + b_in) @ W_out + b_out)   # time_mlp
adarms_cond = time_emb                                              # [B, D]
```

传入 `gemma_expert.model.forward(..., adarms_cond=adarms_cond)`，每层 decoder：

```
residual = x
x, gate = input_layernorm(x, adarms_cond)     ← 5.1 节代码
x = self_attn(x)                              # + prefix KV, RoPE, mask
x = x_residual + x * gate                     # gated residual

residual = x
x, gate = post_attention_layernorm(x, adarms_cond)
x = mlp(x)
x = x_residual + x * gate
```

**单层 denoise 数据流**：

```
x_t → action_in_proj → hidden [B, H, D]
                          │
adarms_cond [B, D] ───────┼──► input_layernorm  (scale/shift/gate)
                          ▼
                     Self-Attention
                          │
                     residual + attn_out * gate
                          │
adarms_cond ──────────────┼──► post_attention_layernorm
                          ▼
                        FFN
                          │
                     residual + ffn_out * gate
                          │
                     (下一层，共 18 层)
                          │
                     final norm(adarms_cond)
                          │
                     action_out_proj → v_t
```

要点：

- `adarms_cond` 是 **序列级/global** 条件（`[B, D]`），经 `unsqueeze(1)` 广播到 H 个 action token
- 同一步内所有 token 共享同一个 `t`；**每层有独立 `dense` 权重**，学到不同的 scale/shift/gate
- 时间条件 **不进入 token 序列、不参与 attention 矩阵**，只调制 Norm 与 residual

### 5.4 与 Pi0 对比

| | **Pi0** | **Pi0.5** |
|---|---------|-----------|
| 时间注入方式 | time_emb concat 进 action token → MLP | adaRMSNorm（Norm + gated residual） |
| `GemmaRMSNorm` | 普通 RMSNorm（`cond=None`，可学习 `weight`） | `use_adarms=True`，`dense(cond)` |
| `adarms_cond` | 无 | 每 denoise 步更新 |
| 设计范式 | token 级融合 | **adaLN-Zero** 风格（DiT / Flow Matching 常见） |

### 5.5 离线预计算与 FlashRT 对应

固定 `num_steps=10` 时，`adarms_cond` 仅依赖 `t`，可离线预算；FlashRT 进一步把 **`dense(adarms_cond)` 也折叠进静态表**：

```python
# precompute.py — 每步每层
time_emb = time_mlp(sincos(t))
sa[s,l] = time_emb @ input_layernorm.dense.weight.T + bias    # attn 前
sf[s,l] = time_emb @ post_attention_layernorm.dense.weight.T + bias  # FFN 前
fs[s]   = time_emb @ final_norm.dense.weight.T + bias
```

推理时 FlashRT C1 `fused_adarms_fp8_static_fp16` 直接读 `sa_ptr`，语义等价于 PyTorch 5.1 节的 `RMSNorm(x)*(1+scale)+shift`；C4/C7 `gate_res_adarms` 等价于 `x + y * gate`。

PyTorch eager 路径仍在每步、每层在线执行 `dense(cond)`；这是相对 FlashRT 的可优化冗余（见第 4.3 节 L2/L3）。

### 5.6 与其他条件化机制的关系

| 机制 | 编码对象 | 注入位置 |
|------|----------|----------|
| **adaRMS** | Flow matching 时间 `t` | Norm 层 + gated residual |
| **RoPE** | token 全局绝对位置 | Attention 的 Q/K |
| **attention mask** | 可见性结构 | Attention softmax |
| **prefix KV** | 图像 + 语言（含 state 文本） | Attention 的 K/V |

**RoPE 管「在哪」；adaRMS 管「去噪到哪一步」** — 两者正交，在 denoise 中同时生效。

---

## 6. Attention 掩码

### 6.1 构造流程（`denoise_step`）

```python
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(B, suffix_len, prefix_len)
suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
full_att_2d_masks   = cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
full_att_2d_masks_4d  = where(full_att_2d_masks, 0.0, -2.38e38)   # [B, 1, Q, K]
```

`make_att_2d_masks`（big_vision 风格 block attention）：

```python
cumsum = cumsum(att_masks, dim=1)
att_2d = (cumsum[:, None, :] <= cumsum[:, :, None]) & pad_2d
```

Query 只能 attend 到 `cumsum(key) <= cumsum(query)` 的有效 key。

### 6.2 Suffix 能看到什么

#### Prefix 侧（所有 suffix token 一致）

- ✅ 全部有效 **image tokens**
- ✅ 全部有效 **language tokens**（Pi0.5 含离散 state 文本）
- ❌ prefix **padding**

#### Suffix 内部

**Pi0.5** — `att_masks = [1, 0, 0, …, 0]`，`cumsum` 全为 1：

- H 个 action token **彼此全互看**（block 内双向 all-to-all，非 causal）

**Pi0** — `att_masks = [1, 1, 0, …, 0]`（含 state token）：

| Query | Suffix 内可见 |
|-------|--------------|
| state_token | 仅自己 |
| action_token | state + 全部 action |

#### 反向：Prefix 看不到 Suffix

Prefix `att_masks` 全 0（cumsum=0），suffix 从 `att_mask=1` 起新 block → prefix **不能** attend action。推理时 prefix 先算完缓存，天然保证单向 conditioning。

### 6.3 与 Prefix 内部 attention 的对比

| | Prefix block | Action block (Pi0.5) |
|---|-------------|----------------------|
| block 内 | 双向 all-to-all | 双向 all-to-all |
| 机制 | 同一 `make_att_2d_masks` | 同一机制 |
| cumsum | 全 0 | 全 1 |
| 跨 block | suffix 可读 prefix；prefix 不可读 suffix | 同左 |

Action chunk 内 attention 与 prefix 内 image↔language 互看 **机制相同**，区别仅在 block 边界与 cumsum 值。

---

## 7. RoPE 位置编码

### 7.1 `rotary_emb(hidden_states, position_ids)`

在 Gemma `forward` 中，**所有 decoder layer 之前**预计算 `(cos, sin)` 并传入各层：

```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)
# 每层 attention:
query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)
```

RoPE 只作用于 **Q 和 K**（V 不变），编码 token 的 **全局绝对位置**。

### 7.2 Pi0.5 denoise 中的 `position_ids`

```python
prefix_offsets = sum(prefix_pad_masks, dim=-1)           # 有效 prefix 长度 L
position_ids   = prefix_offsets + cumsum(suffix_pad_masks) - 1
```

示例：prefix 有效长度 968、action_horizon=10 → suffix 位置 `[968, 969, …, 977]`。

### 7.3 与 KV cache 配合

Denoise 时 `use_cache=False`，attention 拼接 cached prefix K/V 与当前 suffix K/V：

- **Prefix K**：prefix 阶段已用 `position_ids = 0…L-1` 施加 RoPE，写入 cache
- **Suffix Q/K**：本步用 `position_ids = L…L+H-1` 现算 RoPE
- 全局坐标系一致，cross-attention 位置关系正确

---

## 8. Pi0.5 `denoise_step` 逐步拆解

### 8.1 Suffix 嵌入

```
x_t [B,H,action_dim] ──action_in_proj──► action_emb [B,H,D]
timestep t ──sincos + time_mlp──► adarms_cond [B,D]
```

无 state token；time 经 adaRMS 注入，不进 attention 矩阵。

### 8.2 Action Expert Forward

```
gemma_expert.model.forward(
    inputs_embeds     = suffix_embs,
    past_key_values   = prefix KV cache,
    position_ids      = L + [0..H-1],
    attention_mask    = full_att_2d_masks_4d,
    adarms_cond       = adarms_cond,
    use_cache         = False,
)
```

每层：

1. `adaRMSNorm(x, adarms_cond)` — 时间条件 scale/shift/gate（详见 **第 5 章**）
2. Q, K, V = Linear(x)
3. RoPE(Q, K) — 全局位置
4. K = cat(prefix_K_cached, K_suffix)；V 同理
5. Attention + mask — suffix 读 prefix + action 互看
6. FFN + gated residual

### 8.3 输出与 Euler 更新

```
suffix_out[:, -action_horizon:] ──action_out_proj──► v_t [B,H,action_dim]
x_t = x_t + dt * v_t
```

---

## 9. 概念对照表

| 概念 | 作用 | Denoise 每步是否变化 |
|------|------|---------------------|
| `x_t` | 当前 noisy action | ✅ |
| `timestep t` | Flow matching 时间 | ✅（10 个固定值） |
| `adarms_cond` | 时间 → adaRMS 调制 | ✅（可离线预计算） |
| `position_ids` / RoPE | token 空间绝对位置 | ❌（prefix 长度固定时 suffix 位置固定） |
| `full_att_2d_masks` | 谁能 attend 谁 | ❌（结构固定） |
| `past_key_values` | prefix 观测条件 | ❌（每帧算一次） |

**RoPE 管「在哪」；adaRMS 管「去噪到哪一步」** — 两者正交。

---

## 10. 完整数据流图

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段 1：Prefix（每帧 1 次）                                    │
│                                                              │
│  Image(s) ──SigLIP──┐                                        │
│  Lang (+ State 文本) ─┴─► prefix_embs                        │
│           │                                                  │
│           ▼                                                  │
│  PaliGemma LM (2B)                                           │
│    RoPE(pos = 0 … L-1)                                       │
│    → past_key_values                                         │
└──────────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       │ 阶段 2：Denoise × num_steps            │
       │                                       │
       │  x_t (noise → action)                 │
       │  t  (1.0 → 0.1)                       │
       │    │                                  │
       │    ├─ action_in_proj → suffix_emb    │
       │    ├─ sincos(t) → time_mlp → adaRMS  │
       │    │                                  │
       │    ▼                                  │
       │  Gemma Expert (300M)                   │
       │    adaRMSNorm(cond=t)                  │
       │    RoPE(pos = L … L+H-1)              │
       │    Attn: [prefix_KV | suffix_KV]      │
       │      mask: suffix↔prefix + action↔action│
       │    │                                  │
       │    ▼                                  │
       │  v_t = action_out_proj(out)           │
       │  x_t += dt * v_t                      │
       └───────────────────────────────────────┘
                           │
                           ▼
                  actions [B, H, action_dim]
```

---

## 11. FlashRT 对应关系（简要）

FlashRT denoise 管线与 OpenPI 语义对齐，主要差异在算子融合与预计算：

| OpenPI (`pi0_pytorch.py`) | FlashRT |
|---------------------------|---------|
| `action_in_proj(noisy_actions)` | C0 `gmm_fp16` + `add_bias_fp16` |
| `sincos + time_mlp + dense` → adaRMS | 预计算 `sa/sf/fs`（`precompute_adarms_styles`） |
| `apply_rotary_pos_emb` on suffix Q/K | C2b `qkv_split_rope_kvcache_fp16` + 预计算 `dec_rope` |
| prefix KV from TRT/PyTorch LLM | `fill_prefix_kv_from_trt`（K 需 pair-interleave 对齐 RoPE 布局） |
| Euler `x_t += dt * v_t` | 管线末尾或 host 侧 |

详见 `docs/optimizer/flashrt/fusion_design.md` 与 `src/model_optimizer/infer/native/flashrt_decoder/pipeline.py` 内注释。
