# Decoder RoPE 表（`build_dec_rope`）

> 实现：`src/model_optimizer/infer/native/flashrt_decoder/precompute.py::build_dec_rope`  
> 调用：`FlashRtDecoderBackend.setup_prompt`（`backend.py`）  
> 消费：`pipeline.decoder_forward` → `qkv_split_rope_kvcache_fp16`（`weights['rope']`）

---

## 1. 功能概述

`build_dec_rope` 为 **denoise 阶段 action suffix** 预计算一张 **RoPE（Rotary Position Embedding）查找表**，
形状 `[Sa, head_dim]`（Pi0.5 典型为 `[10, 256]`，fp16）。

该表在 FlashRT decoder 中作为 `weights['rope']` 传入 CUDA kernel
`qkv_split_rope_kvcache_fp16`：在 **Q/K 写入 KV cache 时**对 suffix 的 Q、K 施加旋转位置编码。

**它不是训练权重**，只由几何参数（`enc_seq`、`Sa`、`head_dim`）决定，与 checkpoint 无关。

---

## 2. 在整条链路中的位置

```text
Prefix（vit + llm，TRT）
  → past_keys / past_values，序列长度 = enc_seq
  → 填入 Kc/Vc 的前 enc_seq 段（fill_prefix_kv_from_trt）

Denoise（仓内 FlashRT decoder）
  → build_dec_rope(enc_seq, Sa)     ← 本函数
  → 每层 QKV：qkv_split_rope_kvcache_fp16(qkv, rope, …)
     对 Q/K 施 RoPE，写入 Kc/Vc 的 [enc_seq : enc_seq+Sa)
  → attention 读完整 Kc/Vc（prefix + suffix）
```

若 RoPE **起始位置**或**交错布局**与训练/PyTorch 路径不一致，cross-attention 会偏甚至完全错误，
这也是混合 TRT prefix KV + FlashRT denoise 时需在 Thor 上验证 §8.2 KV/RoPE 一致性的原因之一。

---

## 3. 计算步骤（与 FlashRT `pi05_thor.py` 一致）

1. **频率**（Gemma 常用 RoPE，`base=10000`）  
   `inv_freq[d] = 1 / rope_base^(2d/head_dim)`，`d = 0 .. head_dim/2 - 1`。

2. **全序列相位**  
   对位置 `p = 0 .. max_pos-1`：`angle[p,d] = p * inv_freq[d]`，再算 `cos`、`sin`。

3. **只取 suffix 对应的全局位置**  
   - `enc_seq`：prefix（图像 + 语言）token 数，来自 TRT LLM 的 `past_keys.shape[-2]`。  
   - `Sa`：action suffix 长度（`action_horizon`，通常 10）。  
   - 切片 **`kp[enc_seq : enc_seq+Sa]`**：suffix 在**整条序列**中的下标为
     `[enc_seq, enc_seq+Sa)`，不是从 0 重新编号。

4. **交错布局**  
   每个频率维一对 `(cos_d, sin_d)` 拼成  
   `[cos0, sin0, cos1, sin1, ...]`，总长 `head_dim`（256），与 FlashRT kernel 读表约定一致。

---

## 4. 与 KV cache 的对应关系

| 概念 | 含义 |
|------|------|
| `enc_seq` | prefix 长度；RoPE 表从该全局位置起算 |
| `Sa` | suffix（action）token 数 |
| `total_keys` | `enc_seq + Sa`；`Kc/Vc` 物理长度 |
| `kv_offset`（pipeline） | `l * total_keys * HD + enc_seq * HD`；suffix KV 写入起点 |

`build_dec_rope` 的位置切片与 `kv_offset` 中的 `enc_seq` **必须一致**。

---

## 5. 参数说明

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `enc_seq` | ~818（libero） | 当前 prompt 的 prefix 序列长度 |
| `Sa` | 10 | action token 数（`config.action_horizon`） |
| `head_dim` | 256 | 单 KV head 维度（Pi0.5 decoder） |
| `rope_base` | 10000 | RoPE 底数，与 Gemma/PaliGemma 一致 |
| `max_pos` | 1200 | 预计算的最大位置数；需满足 `max_pos >= enc_seq + Sa` |
| `device` | `cuda` | 表驻留设备，与 decoder buffer 一致 |

---

## 6. 调用时机与缓存

- **何时调用**：`FlashRtDecoderBackend.setup_prompt(enc_seq)`，在跑完 prefix、得到 `enc_seq` 之后、
  `backend.run(past_keys, past_values, noise)` 之前。
- **何时重建**：`enc_seq` 变化（不同 prompt、不同图像/语言长度）时必须重建；同一 `enc_seq` 可复用
  `backend` 内已缓存的 `_rope` / `_loop`。
- **与 AdaRMS 预计算**：`precompute_adarms_styles` 并行完成；RoPE 管**位置**，AdaRMS 管
  **flow-matching 时间步调制**（`sa`/`sf`/`fs`）。AdaRMS 与 C1/C4 融合边界见
  [`fusion_design.md`](fusion_design.md)。

---

## 7. 相关文件

| 文件 | 角色 |
|------|------|
| `precompute.py` | `build_dec_rope` 实现 |
| `backend.py` | `setup_prompt` 调用并挂到 `DecoderWeights.rope` |
| `pipeline.py` | `decoder_forward` 中 `qkv_split_rope_kvcache_fp16` |
| `driver.py` | `fill_prefix_kv_from_trt`（prefix KV，与 suffix RoPE 分工） |
| [`fusion_design.md`](fusion_design.md) | 融合算子选取原则、C1/C4 gate 与 OpenPI 对照 |
| [`fp8_gemm_descale_fp16.md`](fp8_gemm_descale_fp16.md) | C2 QKV FP8 GEMM + descale → fp16 `qkv` |
| `docs/optimizer/ddup/native_decoder_implementation_todo.md` §8.2 | KV/RoPE 契约与 Thor 验证要点 |

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-02 | 初版：从 `build_dec_rope` 代码审阅整理 |
