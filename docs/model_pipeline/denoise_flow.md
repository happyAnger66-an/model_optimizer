# Pi0.5 Denoise 流程详解

本文档聚焦 **Pi0.5 的 denoise（Flow Matching 采样）过程**：从整体架构与 time / state / action 三类信号如何进入模型、如何作用，到 AdaRMSNorm 等实现细节。

更完整的推理管线（WebUI、FlashRT、TRT）见 [`pipeline.md`](pipeline.md)。

**主要代码**：

- `third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py` — `sample_actions` / `denoise_step` / `embed_suffix`
- `third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` — AdaRMSNorm / gated residual / RoPE
- `third_party/openpi/src/openpi/models/tokenizer.py` — state 离散化进 prompt

---

# 第一部分：整体架构

## 1.1 模型范式：Flow Matching + DiT 风格条件化

Pi0.5 denoise **不是 DDPM 式 score-based diffusion**，而是 **Conditional Flow Matching**：

| 层次 | 是什么 | Pi0.5 做法 |
|------|--------|-----------|
| **生成范式** | Flow Matching | 线性路径 + 速度场预测 + Euler ODE 采样 |
| **速度场网络** | Transformer | PaliGemma prefix + Gemma action expert suffix |
| **时间条件化** | DiT 式 adaLN-Zero | **adaRMSNorm**（scale/shift/gate，零初始化 dense） |

官方配置（`pi0_config.py`）明确：`action expert uses adaRMSNorm to inject the **flow matching timestep**`。

更准确的说法：**Flow Matching VLA，action expert 采用 DiT 风格 adaRMS 注入时间步**——不是图像领域的标准 DiT 模型。

---

## 1.2 三类信号及其分工

Denoise 过程中，模型同时处理三种正交信息：

| 信号 | 含义 | 进入模型的方式 | 在 denoise 中是否每步变化 |
|------|------|----------------|-------------------------|
| **观测（含 state）** | 图像 + 任务语言 + 机器人状态 | **Prefix**：SigLIP + 语言 token（state 离散化写入 prompt） | ❌ 每帧算一次 KV cache |
| **时间 `t`** | Flow matching 去噪进度（1→0） | **adaRMS**：sincos → time_mlp → `adarms_cond` | ✅ 每 denoise 步不同 |
| **Action `x_t`** | 当前 noisy action chunk | **Suffix**：`action_in_proj(x_t)` → action token | ✅ Euler 每步更新 |

```
                    ┌─── 观测条件（Prefix，cached）──────────────┐
                    │  Image ──SigLIP──┐                         │
                    │  Task prompt     ├──► PaliGemma LM ──► KV │
                    │  State (离散文本)─┘                         │
                    └────────────────────────────────────────────┘
                                      ▲ cross-attn（每步可读）
                    ┌─────────────────┴──────────────────────────┐
  x_t ──action_in_proj──► action tokens [B, H, D]                │
  t  ──sincos+time_mlp──► adarms_cond [B, D] ──► adaRMS 各层    │
                    │         Gemma Expert (300M) × 18 layers     │
                    └────────────────────────────────────────────┘
                                      │
                              v_t = action_out_proj(out)
                                      │
                              x_t ← x_t + dt * v_t
```

**RoPE** 编码 token **空间位置**（prefix 后连续编号）；**adaRMS** 编码 **去噪时间**——两者正交。

---

## 1.3 State：不在 suffix，而在 prefix 语言里

Pi0.5 与 Pi0 的关键区别（`Pi0Config` 注释）：

- Pi0：state → `state_proj` → suffix 第 1 个连续 token
- **Pi0.5**：state → 256-bin 离散化 → 拼进语言 prompt

```python
# tokenizer.py — Pi0.5 格式
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

`denoise_step(..., state, ...)` 和 `embed_suffix(state, ...)` **仍接收 state 参数，但 Pi0.5 下不使用**（API 兼容）。State 通过 prefix 语言 token 进入 PaliGemma KV cache，suffix action token 经 attention **读取**这些 prefix key。

推理须保证 `TokenizePrompt(discrete_state_input=True)` 正确执行。

---

## 1.4 Flow Matching：训练目标与推理采样

### 训练（`PI0Pytorch.forward`）

```python
time ~ Beta(1.5, 1) * 0.999 + 0.001          # 随机 t ∈ (0, 1)
x_t = t * noise + (1 - t) * actions            # 线性插值路径
u_t = noise - actions                          # 目标速度场（路径常数）
v_t = action_out_proj(expert(suffix(x_t), t))  # 模型预测
loss = MSE(v_t, u_t)
```

### 推理（`sample_actions`）

```python
dt = -1.0 / num_steps     # 默认 10 步 → dt = -0.1
x_t = noise               # t = 1，纯噪声
time = 1.0
while time >= -dt / 2:    # t = 1.0, 0.9, …, 0.1
    v_t = denoise_step(obs, x_t, time)
    x_t = x_t + dt * v_t  # Euler 积分
    time += dt
return x_t                # 最终 action chunk
```

- **约定**：`t=1` 为噪声，`t=0` 为干净 action（与部分 diffusion 文献相反）
- 每步 `denoise_step` 预测速度场 `v_t`；`x_t + dt·v_t` 沿 ODE 从噪声走向 action

---

## 1.5 两阶段推理结构

```
阶段 1 — Prefix（每帧 1 次）
  embed_prefix → PaliGemma LM(use_cache=True) → past_key_values

阶段 2 — Denoise 循环（每帧 num_steps 次，默认 10）
  for t in [1.0, 0.9, …, 0.1]:
      v_t = denoise_step(x_t, t, past_key_values)
      x_t = x_t + dt * v_t
```

| 组件 | 模型 | 频率 |
|------|------|------|
| Prefix | PaliGemma 2B language model | 1× / 帧 |
| Denoise | Gemma 300M action expert | 10× / 帧 |

Prefix KV 在整个 denoise 循环中 **只读复用**；每步仅重算 suffix（action tokens + 当前 `t` 的条件化）。

---

## 1.6 单步 `denoise_step` 总览

```python
# pi0_pytorch.py — denoise_step 逻辑摘要
suffix_embs, ..., adarms_cond = embed_suffix(state, x_t, timestep)  # action + time

# attention mask: suffix 可读 prefix + action 互看
full_att_2d_masks_4d = prepare(full_att_2d_masks)
position_ids = prefix_len + suffix_positions

suffix_out = gemma_expert.forward(
    inputs_embeds=suffix_embs,
    past_key_values=past_key_values,      # prefix KV
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    adarms_cond=adarms_cond,
)
v_t = action_out_proj(suffix_out[:, -action_horizon:])
return v_t
```

---

# 第二部分：实现细节

## 2.1 Action `x_t`：Suffix 嵌入

```python
action_emb = action_in_proj(x_t)    # [B, H, action_dim] → [B, H, D]
# H = action_horizon, D = expert hidden size
```

- `x_t` 是当前 noisy action chunk，每 Euler 步更新
- 经线性层抬升到 expert 宽度，作为 suffix 唯一 token 序列（Pi0.5 无 state token）
- **不**与 time 拼接（Pi0 才 concat + MLP）

---

## 2.2 时间 `t`：sincos → time_mlp → adarms_cond

```python
time_emb = create_sinusoidal_pos_embedding(t, D, min_period=4e-3, max_period=4.0)
time_emb = silu(silu(time_emb @ W_in + b_in) @ W_out + b_out)   # time_mlp
adarms_cond = time_emb    # [B, D]
```

- `t` **不是** attention 里的 token；通过 adaRMS 注入各层 Norm 与 residual
- 固定 `num_steps=10` 时，`adarms_cond` 仅依赖 `t`，可离线预计算（FlashRT `sa/sf/fs`）

### sincos 编码原理

在 `[4e-3, 4.0]` 上 log 均匀取 `D/2` 个周期，多频率 sin/cos 拼接为 `D` 维向量，使模型在 \(t \in [0,1]\) 多尺度分辨去噪进度。

---

## 2.3 AdaRMSNorm 详解

Pi0.5 action expert 配置 `use_adarms=[False, True]`：仅 expert 启用，`GemmaRMSNorm` 带 `dense(cond)`。

### 核心代码

```python
# GemmaRMSNorm.forward — Pi0.5 路径
normed = RMSNorm(x)                              # _norm(x)
modulation = dense(adarms_cond)                  # [B, D] → [B, 3D]
modulation = modulation.unsqueeze(1)             # → [B, 1, 3D]，广播到 H 个 token
scale, shift, gate = chunk(modulation, 3)

out = normed * (1 + scale) + shift               # 条件化仿射
return out, gate
```

| 输出 | 作用 | 使用位置 |
|------|------|----------|
| **scale, shift** | 调制 Norm 后的特征 | Attention / FFN 的 **输入** |
| **gate** | 缩放 sublayer 输出 | **gated residual** 加回时 |

`dense` **零初始化**：训练初期 scale≈0、shift≈0、gate≈0，接近恒等映射（adaLN-Zero 思路）。

### 每层 decoder 数据流（Attention 分支）

```python
residual = x                                           # 357: 保存 skip
x_norm, gate = input_layernorm(x, adarms_cond)         # 358: adaRMS(x, t)
attn_out = self_attn(x_norm)                           # 361: 注意输入是 x_norm，不是 x
out = residual + attn_out * gate                         # 372: gated residual
```

对比：

| 公式 | 场景 |
|------|------|
| `x + attn(norm(x))` | 标准 Pre-Norm（Pi0 expert / gate=None） |
| `x + attn(ada_norm(x,t)) * gate(t)` | **Pi0.5** |

FFN 分支（375–378 行）结构相同，gate 来自 `post_attention_layernorm`。

### 与 Pi0 时间注入对比

| | Pi0 | Pi0.5 |
|---|-----|-------|
| 时间进入 token | sincos(t) concat action → MLP | ❌ |
| 时间进入 Norm | ❌ | **adaRMS** scale/shift |
| 时间进入 residual | ❌ | **gate** |

---

## 2.4 Gated Residual

```python
def _gated_residual(x, y, gate):
    if gate is None:
        return x + y
    return x + y * gate
```

- **372 行**：Attention 后 — `x + attn_out * gate_attn`
- **378 行**：FFN 后 — `x + ffn_out * gate_ffn`

gate 与 scale/shift 同源（同一次 `input_layernorm` / `post_attention_layernorm`），但分工不同：scale/shift 调制 **进入 sublayer 的特征**，gate 调制 **sublayer 输出加回 skip 的量**。

---

## 2.5 Attention：Suffix 能看到什么

Mask 构造（`denoise_step`）：

```python
full_att_2d_masks = cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=K)
full_att_2d_masks_4d = where(mask, 0.0, -inf)    # [B, 1, suffix_Q, prefix_K+suffix_K]
```

Pi0.5 suffix `att_masks = [1, 0, 0, …, 0]` → H 个 action token **彼此双向互看**。

| Key 类型 | Suffix 可见？ |
|----------|--------------|
| 有效 image / language tokens（含 state 文本） | ✅ |
| 其他 action tokens | ✅（双向） |
| prefix / suffix padding | ❌ |
| Prefix → Suffix（反向） | ❌（prefix 先缓存，不读 action） |

传入 Gemma 的 `create_causal_mask`：因 mask 已是 4D，**early exit 原样透传**，不叠加默认 causal mask。

---

## 2.6 RoPE 位置编码

```python
position_ids = prefix_offsets + cumsum(suffix_pad_masks) - 1
# 例：prefix 长 968 → suffix 位置 [968, 969, …, 977]
```

- Prefix KV：prefix 阶段已用 `position_ids = 0…L-1` 施加 RoPE 并缓存
- Suffix Q/K：denoise 每步用全局位置 `L…L+H-1` 现算 RoPE
- Attention 拼接 `[prefix_K_cached | suffix_K]`，全局坐标一致

**RoPE 管「在哪」；adaRMS 管「去噪到哪一步」。**

---

## 2.7 条件化机制对照

| 机制 | 编码对象 | 注入位置 | 每 denoise 步变化 |
|------|----------|----------|------------------|
| Prefix KV | 图像 + 语言 + state | Attention K/V | ❌ |
| **adaRMS** | Flow time `t` | Norm + gated residual | ✅ |
| RoPE | token 绝对位置 | Attention Q/K | ❌（prefix 长度固定） |
| Attention mask | 可见性结构 | Attention softmax | ❌ |
| `x_t` | noisy action | Suffix embedding | ✅ |

---

## 2.8 离线预计算（FlashRT）

固定 `num_steps=10` 时，时间相关计算可全部折叠：

```
t → sincos → time_mlp → adarms_cond
  → 每层 dense(adarms_cond) → sa / sf / fs（scale/shift 预 baked）
  → kernel: RMSNorm + scale/shift；residual + out * gate
```

详见 `src/model_optimizer/infer/native/flashrt_decoder/precompute.py` 与 [`pipeline.md` §11](pipeline.md)。

---

# 附录：完整 Denoise 数据流

```
┌──────────────────────────────────────────────────────────────┐
│ 输入                                                          │
│  observation: images, prompt(+discrete state), state tensor  │
│  noise ~ N(0,I): [B, H, action_dim]                          │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Prefix（1×/帧）─────────────────────────────────────────────┐
│  embed_prefix → PaliGemma LM → past_key_values               │
│  RoPE(pos = 0 … L-1)                                         │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Denoise × num_steps ────────────────────────────────────────┐
│  x_t ← noise;  t ← 1.0;  dt ← -1/num_steps                 │
│  loop while t >= -dt/2:                                      │
│    ┌─ embed_suffix ─────────────────────────────────────┐    │
│    │  x_t → action_in_proj → suffix_emb [B,H,D]       │    │
│    │  t  → sincos → time_mlp → adarms_cond [B,D]      │    │
│    └───────────────────────────────────────────────────┘    │
│    ┌─ gemma_expert × 18 layers ────────────────────────┐    │
│    │  adaRMSNorm(x, adarms_cond) → scale/shift/gate     │    │
│    │  Self-Attn(norm(x), prefix_KV, RoPE, mask)        │    │
│    │  x + attn_out * gate                               │    │
│    │  adaRMSNorm → FFN → x + ffn_out * gate             │    │
│    └───────────────────────────────────────────────────┘    │
│    v_t = action_out_proj(suffix_out)                         │
│    x_t = x_t + dt * v_t;  t = t + dt                        │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
   actions [B, H, action_dim]
```
