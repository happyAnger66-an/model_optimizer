# Pi0.5 参数量分析

本文档总结 **Pi0.5（`pi05_libero` 配置）** 的参数量拆解，覆盖 **ViT + LLM + Action + Embedding** 四大模块，并说明 checkpoint 全量与 VLA 推理活跃参数的差异。

配置来源：

- `third_party/openpi/src/openpi/models/pi0_config.py`
- `third_party/openpi/src/openpi/models/gemma.py`
- `third_party/openpi/examples/convert_jax_model_to_pytorch.py`

---

## 1. 架构配置

| 子模块 | 变体 | hidden | layers | FFN | heads / KV | head_dim |
|--------|------|--------|--------|-----|------------|----------|
| **ViT (SigLIP)** | SigLIP-L | 1152 | 27 | 4304 | 16 / 16 | 72 |
| **LLM (PaliGemma)** | `gemma_2b` | 2048 | 18 | 16384 | 8 / 1 (GQA) | 256 |
| **Action Expert** | `gemma_300m` | 1024 | 18 | 4096 | 8 / 1 (GQA) | 256 |
| **Embedding** | PaliGemma | vocab=257152 | — | — | — | dim=2048 |
| **Action I/O** | Pi0.5 特有 | 32 ↔ 1024 | — | — | — | — |

Pi0.5 相对 Pi0 的关键差异：

- `use_adarms=[False, True]`：仅 **Action Expert** 启用 AdaRMSNorm，LLM prefix 不用
- `time_mlp` 将 Flow Matching 时间步编码为 `adarms_cond`，注入 Expert 各层 Norm
- `action_horizon=10`、`action_dim=32`（`pi05_libero`）**不改变参数量**，只影响运行时 shape 和 denoise 循环次数

---

## 2. 参数量总览

| 模块 | 参数量 | 占推理活跃参数比例 | VLA 推理是否使用 |
|------|--------|-------------------|------------------|
| **ViT** | **~415M** | 12.4% | ✅ |
| **Embedding** | **~527M** | 15.7% | ✅ |
| **LLM** | **~1,982M** | 59.1% | ✅ |
| **Action** | **~430M** | 12.8% | ✅ |
| **推理活跃合计** | **~3.35B** | 100% | — |
| lm_head（PaliGemma + Expert） | **~790M** | — | ❌ 不参与 VLA 前向 |
| **Checkpoint 全量** | **~4.14B** | — | 含未使用的 lm_head |

> **VLA 实际推理用到的参数量约 3.35B。** Checkpoint 全量约 4.14B，多出的 ~790M 为两个 `lm_head`（下文称「死重」）。

---

## 3. 各模块拆解

### 3.1 ViT（~415M）

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Patch + Position Embed | ~1.0M | `14×14×3 → 1152`，256 个位置编码 |
| SigLIP Encoder（27 层） | ~411.5M | 主体，占 ViT 99% |
| Post LayerNorm | ~2.3K | 可忽略 |
| Multi-modal Projector | ~2.4M | `1152 → 2048`，对齐 LLM hidden |

SigLIP 单层（含 bias）约 **15.2M**，27 层合计约 **411M**。

### 3.2 Embedding（~527M）

| 组件 | 参数量 | 说明 |
|------|--------|------|
| `embed_tokens` | **526,647,296** | `257152 × 2048` |

语言 token（含 prompt 中离散化 state 文本）通过该表查表得到 embedding。它独立于 ViT，也不属于 LLM 的 Transformer 层权重。

### 3.3 LLM（~1,982M）

PaliGemma **Language Model**（18 层 Gemma，GQA）：

| 组件 | 参数量 | VLA 推理 |
|------|--------|----------|
| 18 层 Transformer | ~1,981.9M | ✅ prefix 编码，写 KV cache |
| Final RMSNorm | ~2K | ✅ |
| `lm_head` | ~527M | ❌ 文本生成头，VLA 不用 |

单层 Gemma 2B（无 bias）约 **109.1M**：

- Attention：`2×D² + 2×D×256` ≈ 9.4M
- MLP：`3×D×16384` ≈ 100.7M

### 3.4 Action（~430M）

| 组件 | 参数量 | VLA 推理 |
|------|--------|----------|
| Gemma Expert 18L（含 AdaRMS） | ~427.9M | ✅ denoise × `num_steps`（默认 10） |
| `action_in_proj` | ~33K | `32 → 1024` |
| `action_out_proj` | ~33K | `1024 → 32` |
| `time_mlp_in` / `time_mlp_out` | ~2.1M | 时间 → AdaRMS 条件向量 |
| Expert `lm_head` | ~263M | ❌ 不用，输出走 `action_out_proj` |

**Action ≠ 裸 `gemma_300m`：** openpi 所称的 `gemma_300m` backbone（无 AdaRMS）约 **311M**；Pi0.5 的 AdaRMS 约额外增加 **~116M**（见第 4 节）。

---

## 4. AdaRMS 的 ~116M 是什么

这 **~116M 不是 `time_mlp` 本身**，而是 **Action Expert 各层 AdaRMSNorm 的 `dense` 调制层**：

```text
每层 2 个 Norm（attn 前 + FFN 前）
  dense: cond_dim(1024) → 3×hidden(3072)   # 输出 scale / shift / gate

18 层 × 2 × (1024×3072)  ≈ 113M
+ final norm 的 dense      ≈   3M
─────────────────────────────────
合计                       ≈ 116M
```

时间条件注入路径：

```text
flow matching 时间 t
  → sincos 位置编码
  → time_mlp_in / time_mlp_out   (~2.1M，把 t 编成 cond 向量)
  → adarms_cond [B, 1024]
  → 每层 GemmaRMSNorm(cond=adarms_cond)
       dense(cond) → scale, shift, gate
       对 hidden states 做自适应归一化
```

| 组件 | 参数量 | 角色 |
|------|--------|------|
| `time_mlp` | ~2.1M | 把标量时间 **编码** 成 cond 向量 |
| AdaRMS `dense`（18L×2 + final） | ~116M | 把 cond **注入** 各层 Norm，控制去噪步行为 |

范围限定：

- **只在 Action Expert 上**：`use_adarms=[False, True]`，LLM prefix 无 AdaRMS
- **只在 denoise 阶段使用**：prefix 编码不走此路径
- 每个 denoise step 的 `t` 不同，故每步 `adarms_cond` 不同（FlashRT 可预计算为 `sa/sf/fs`）

---

## 5. 「死重」是什么意思

**死重**指 checkpoint 里保存了、但 **VLA 推理计算图不会调用** 的参数。

| 参数 | 大小 | 为何存在 | VLA 推理 |
|------|------|----------|----------|
| `paligemma.lm_head` | ~527M | 语言模型文本生成（预测下一个 token） | ❌ |
| `gemma_expert.lm_head` | ~263M | Expert 侧同类 head | ❌ |

VLA 推理的实际输出路径为：

```text
Expert hidden states → action_out_proj → 32 维 action
```

而非 `lm_head → vocab logits`。

注意：「死重」是 **功能上不参与 VLA 前向**，不等于可以随意删除——若需继续语言预训练/微调，可能仍需保留。部署场景可裁剪以减小 checkpoint 体积。

---

## 6. 推理时的参数归属

```text
┌─────────────────────────────────────────────────────────┐
│ Prefix（每帧 1 次）                                       │
│                                                         │
│  Image → ViT (~415M)                                    │
│       → projector → visual tokens                       │
│  Lang tokens → Embedding (~527M) → token embeddings      │
│       → concat → prefix_embs                            │
│       → LLM 18L (~1,982M) → KV cache                    │
└─────────────────────────────────────────────────────────┘
                          │ past_key_values
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Denoise（每帧 num_steps 次，默认 10）                      │
│                                                         │
│  noise x_t → action_in_proj (~33K)                      │
│  time t → time_mlp (~2.1M) → adarms_cond                │
│       → Expert 18L + AdaRMS (~428M)                      │
│       → action_out_proj (~33K) → v_t                    │
└─────────────────────────────────────────────────────────┘
```

---

## 7. 关键结论

1. **VLA 推理活跃参数约 3.35B**，最大头是 **LLM（~2.0B，59%）**，其次 **Embedding（~527M）** 和 **Action Expert（~430M）**，**ViT（~415M）** 约占 12%。
2. **Checkpoint 全量约 4.14B**，多出的 **~790M** 为两个 `lm_head`，不参与 VLA 前向。
3. **Pi0.5 的 AdaRMS 约 +116M**，是 Expert 各层 time-conditioned Norm 的 `dense` 调制层；`time_mlp`（~2M）负责生成条件向量。
4. `pi05_libero` 的 `action_horizon` / `action_dim` 只影响运行时，不改变参数量。

---

## 8. 计算方法说明

参数量由架构维度解析计算（GQA、无 attention bias、SigLIP 含 bias），与 openpi 源码配置一致。若需与磁盘 checkpoint 逐 key 对账，可对 safetensors 按前缀统计：

- `vision_tower.*` → ViT
- `embed_tokens` / `language_model.*` → Embedding + LLM
- `gemma_expert.*` / `action_*` / `time_mlp_*` → Action
