# FlashRT Pi0.5 推理 Pipeline 说明

本文承接 [load_model.md](./load_model.md) 中 **`VLAModel` / Frontend 构造** 的结论，从 **推理执行** 角度说明 Pi0.5 在 FlashRT 中的 **pipeline 如何组装**、**各阶段按层如何选择算子**、以及 **Thor 与 RTX 两条硬件路径的差异**。代码路径以 **FlashRT 仓库根目录** 为基准（`flash_rt/` 包）。

---

## 1. 总览：谁在「拼」Pipeline

| 角色 | 职责 |
|------|------|
| **Frontend**（如 `Pi05TorchFrontendThor` / `Pi05TorchFrontendRtx`） | 加载权重、分配/绑定 GPU 指针、构造 `GemmRunner`（cuBLASLt）、可选注入 **Attention Backend**、把 `infer` 里的图像/噪声/语言嵌入写入 pipeline 缓冲区并触发一次 forward。 |
| **Pipeline** | **编排** `flash_rt_kernels`（`fvk`）里的融合算子 + **GEMM**（`gemm` 或 `fvk.gmm_*`）+ **注意力**（`attn` 或内联 `fvk` 调用），不持有框架张量语义，只认指针与维度。 |
| **Attention Backend** | 把「写 Q/K/V 的缓冲区」与「跑哪条注意力实现」封装起来；**同一套 pipeline 循环** 在每一层调用 `run` / `vision_attn` 等，由 backend 选择底层 kernel。 |

端到端数据流（概念上）：

```text
观测图像 (bf16/fp16)
  → SigLIP（逐层：Norm → FP8 QKV GEMM → 注意力 → O GEMM + FFN）
  → PostLN + 多模态投影 + 语言 token 拼接到 encoder_x
  → PaliGemma Encoder（18 层自注意力 + FFN，写 K/V cache）
  → Gemma Decoder（扩散步 × 18 层：AdaRMS + 交叉注意力读同一 K/V + FFN）
  → 动作 chunk（写在 diffusion 噪声缓冲上，与原噪声 in-place 迭代）
```

---

## 2. 两条硬件主线：Thor vs RTX

### 2.1 Thor（SM110 等）

- **共享视觉 + Encoder**：`flash_rt/hardware/thor/shared_primitives.py` 中的 **`siglip_forward`**、**`postln_project`**、**`encoder_forward`**（Pi0 / Pi0.5 / GROOT 复用）。
- **Pi0.5 专用 Decoder**：`flash_rt/models/pi05/pipeline_thor.py` 中的 **`decoder_forward`**（AdaRMSNorm + 静态 FP8 标量方案；与 Pi0 的 decoder 分离，符合 `adding_new_model.md` 的约定）。
- **GEMM**：大量通过 **`fvk`** 的 FP8/FP16 融合入口（如 `fp8_gemm_descale_fp16`、`gmm_fp16`），与 **CUTLASS/GemmRunner** 的组合见 encoder 注释中的「static FP8」描述。
- **注意力**：可选 **`ThorFlashAttnBackend`**（`flash_rt/hardware/thor/attn_backend.py`），通过 **`attn.run(site, layer_idx, ...)`** 分发；`attn=None` 时走 **`fvk.fmha_strided_full`**（SigLIP）或 **`fvk.attention_qkv_fp16`**（Encoder/Decoder 交叉或自注意力）等直连路径。

### 2.2 RTX（消费级 Blackwell / Ada 等）

- **单类编排**：`flash_rt/models/pi05/pipeline_rtx.py` 中的 **`Pi05Pipeline`**。
- **GEMM**：**`GemmRunner`**（cuBLASLt）负责 BF16/FP8 矩阵乘；大 GEMM 可在 FP8 权重 + 设备侧 activation scale 下走 `fp8_nn_*` 族接口。
- **注意力**：**`RtxFlashAttnBackend`**（`flash_rt/hardware/rtx/attn_backend.py`），默认 **vendored Flash-Attention 2**（`flash_rt.flash_rt_fa2`）；环境变量 **`FVK_RTX_FA2` / `FVK_RTX_FA2_SITES`** 可切到上游 `flash_attn` 或按站点禁用 FA2。
- **图捕获**：**`record_infer_graph`** → 内部 **`run_pipeline`** 被 CUDA Graph 包裹；需与 PyTorch stream 对齐时传入 `external_stream_int`（见该类 docstring）。

---

## 3. RTX：`Pi05Pipeline` 的构建与执行顺序

### 3.1 构造期（`__init__`）

1. 保存 **`gemm` / `fvk` / `attn_backend`** 与权重指针字典（Vision BF16、Vision/Encoder/Decoder FP8 键、Decoder 时间嵌入与 Ada 风格参数等）。
2. 由 **`num_views`、`max_prompt_len`、`chunk_size`、`num_steps`** 推导 **`vision_seq`、`encoder_seq_len`、`total_kv`**（Encoder 序列 + chunk 上的 KV 总长）。
3. **`attn_backend.get_ptrs()`**：取得 **vis_Q/K/V、enc_Q、enc_K/V（按层分层）、dec_Q** 等 **device int 指针**，pipeline 后续用 `fvk` 把 QKV 直接写入这些地址（**零拷贝对齐 FA2 布局**）。
4. **`_allocate_buffers`**：内部工作区全部为 **`CudaBuffer`**（无 torch 张量语义）。
5. **RoPE 表**、**valid_encoder_len**、RMS **ones** 向量、FP8 **activation scratch** 与 **per-op 静态 scale** 缓冲等。
6. 可选：**CUDA Graph** 在 **`record_infer_graph`** 时才捕获，不在 `__init__` 默认完成。

### 3.2 单次推理调用链

```text
set_language_embeds (每 prompt 一次，可选更新 decoder RoPE 切片)
  → forward()
       ├─ 若已 capture：graph.replay
       └─ 否则：run_pipeline(stream)
              ├─ _copy_lang_embeds_to_encoder_x   # 防止 encoder 残差覆盖语言槽
              ├─ vision_encoder
              ├─ transformer_encoder
              └─ transformer_decoder
```

- **`input_images_buf`**：归一化图像。
- **`input_noise_buf`**：扩散初噪声；**结束后同一缓冲即为动作输出**。
- **`set_language_embeds`**：在 device 侧持久保存一份语言嵌入，并在 **每次 `run_pipeline` 开头** 拷回 `encoder_x` 的 `[vision_seq : vision_seq+prompt_len]`，因为 encoder 会 **原地** 改写该段。

### 3.3 FP8 标定、Autotune、Graph 的先后

1. **`calibrate_fp8()`**：跑一次 **`run_pipeline`**，在「未标定」路径下由 **`_fp8_gemm`** 侧写 **activation scales**；完成后 **`fp8_calibrated = True`**，后续可走静态 scale 融合核。
2. **`autotune_gemms()`**：对 **Vision / Encoder / Decoder** 各典型 **M,N,K** 形状调用 **`gemm.autotune_bf16_nn` / `autotune_fp8_nn_dev`**，把最优 **cuBLASLt algorithm** 缓存到 `GemmRunner`。
3. **`record_infer_graph()`**：若未标定会先 **`calibrate_fp8()`**；再 **`autotune_gemms()`**；然后在指定 stream 上 **warmup 3 次** 后 **begin_capture → run_pipeline → end_capture**。

---

## 4. Thor：共享 Primitive + Pi0.5 Decoder

### 4.1 SigLIP：`siglip_forward`

对 **`l in 0..L-1`（L=27）** 每层固定模式（摘要）：

| 步骤 | 算子类型 | 说明 |
|------|----------|------|
| Pre-Attn LN | `fvk.layer_norm_fp8` | 输出进 FP8 激活缓冲 |
| QKV | `gemm.fp8_nn_bias` | FP8 × 权重 + bias → interleaved QKV（fp16） |
| 注意力 | **`attn.run("siglip", 0, q_seq=spv)`** 或 `fvk.fmha_strided_full` | 多视角独立：按 view 批、每 view **256** token；interleaved QKV 用 **stride=3D** 取 Q/K/V 指针 |
| O + Res | `gemm.fp8_nn_bias_res` | 注意力输出量化后再 O 投影并残差 |
| FFN | LN → FP8 → `gemm.fp8_nn_gelu_bias` → Down `fp8_nn_bias_res` | 与 Vision 标准 Transformer FFN 一致 |

**层间无「算子表分支」**：每层 **同一套** 调用序列，区别仅在 **该层权重指针** 与 **alpha 标量**（`weights['alpha'][l*4 + ...]`）。

### 4.2 Encoder：`encoder_forward`

18 层循环；每层典型包含：**RMSNorm→FP8**、**QKV GEMM（带 descale/alpha）**、**RoPE + KV 写入 `Kc`/`Vc`**、**自注意力**（`attn.run("encoder", l, q_seq=Se, ...)` 或 `fvk.attention_qkv_fp16`）、**O**、**Gate+Up / Down（GeGLU 路径）** 等。  
**`total_keys`** 维度预留 **encoder 序列 + decoder chunk**，便于 **decoder 在同一物理 KV 缓冲上追加** action token 的 K/V。

### 4.3 Decoder：`decoder_forward`（`pipeline_thor.py`）

外层 **`for s in range(steps)`**（扩散步），内层 **`for l in range(layers)`**（18 层）。每层子块可概括为：

1. **AdaRMSNorm → FP8**：`fvk.fused_adarms_fp8_static_fp16`（用 **per-layer 静态 act_scale**）。
2. **QKV**：`fvk.fp8_gemm_descale_fp16`（权重切片 `qw + l * D * 2560`）。
3. **RoPE + QKV split + KV cache**：`fvk.qkv_split_rope_kvcache_fp16`，**`kv_offset`** 依赖 **层索引 `l`**、**`enc_seq`**、**`total_keys`**。
4. **交叉注意力**：**`attn.run("decoder", l, q_seq=S, kv_seq=total_keys)`** 或 **`fvk.attention_qkv_fp16`**，K/V 指向 **该层** 在 `Kc`/`Vc` 中的偏移（含 prefix + action tokens）。
5. **O 投影** → **gate×residual + 下一层 AdaRMS**（或最后一步的 `gate_res_fp16`）→ **Gate+Up GEMM** → **SiLU×Up** → **Down GEMM**。
6. 每个扩散 step 末尾：**AdaRMS + action out GEMM** 更新 **`noise`** 缓冲（与训练时的 denoise 调度一致，由权重/前端约定）。

**静态 FP8**：`w_scales` / `act_scales` 按 **`(step, layer, sub-block)`** 展平索引（代码中以 **`(s * layers + l) * S * D3`** 等偏移访问 **style / scale** 向量），保证 **同一套 kernel** 在全程复用，仅指针与标量偏移变化。

---

## 5. 注意力：Backend 如何按「站点 + 层」选 Kernel

### 5.1 Thor：`ThorFlashAttnBackend.run`

- **`site == "siglip"`**：始终 **`fvk.fmha_strided_full`**（与 legacy 路径一致，backend 只统一入口）。
- **`site == "encoder"` / `"decoder"`**：由 **`AttentionSpec` 中 `site.extra["kernel"]`** 决定：
  - **`"standard"`（默认）**：`fvk.attention_qkv_fp16`；
  - **`"state_masked"`**：`fvk.attention_qkv_fp16_state_masked`（供 Pi0 等 **state token** 场景；Pi0.5 默认不用）。

**`make_pi05_attention_spec`**（同文件）固定三站点层数：**SigLIP 27**、**Encoder 18**、**Decoder 18**，并写明 **GQA**：Encoder/Decoder 均为 **8 Q head、1 KV head、head_dim=256**；SigLIP **16×72**、每 view **256** token。

### 5.2 RTX：`AttnBackend` 协议

Pipeline 在 Vision / Encoder / Decoder 各层调用 **`vision_attn`**、**`encoder_attn(layer_idx, seq)`**、**`decoder_attn(layer_idx, enc_seq, dec_seq)`**。  
Backend 内部选择 **bf16/fp16 FA2 入口**（由 dtype 与是否 vendored 决定），并保证 **CUDA Graph 捕获** 下输出指针稳定（vendored 路径 **预分配 O**）。

---

## 6. 「按层选算子」到底选什么

Pi0.5 **没有**「每层从几十种 kernel 动态搜索」的抽象；**结构固定**，**层索引 `l` 只影响**：

1. **权重切片基址**（如 `qw + l * ...`、`Kc + l * stride`）。
2. **标量/风格缓冲偏移**（Thor decoder 的 `sa`/`sf`/`act_scales`/`w_scales`）。
3. **Attention Backend 的 `layer_idx`**（选中 **第 `l` 层** 的 K/V cache 面片）。

**可配置/可选路径**主要体现在：

- **是否注入 `attn`**：Thor 上 legacy **`fvk`** 直连 vs **`ThorFlashAttnBackend`**（行为对齐，便于测试与渐进迁移）。
- **RTX 上 `use_fp8` / `use_fp8_decoder`**：关闭后走 **BF16 `gemm.bf16_nn` + `fvk` 非 FP8 融合分支**（`pipeline_rtx` 内大量 `if self.use_fp8` 分支）。
- **环境变量 `FVK_RTX_FA2*`**：切换 FA2 实现或站点集合。
- **`record_infer_graph` 与否**：不改变算子，只改变 **调度方式**（Graph replay vs 直接 launch）。

**FP4 等变体**（若前端启用 `use_fp4`）在 **另一条 frontend/pipeline** 中绑定不同 `fvk`/权重布局；与本文主线 **BF16+FP8** 并行存在，细节以对应 `*_fp4.py` 与 CMake 开关为准。

---

## 7. 与 `load_model` 的衔接

- [load_model.md](./load_model.md) 说明 **如何根据 `hardware`/`config` 选中哪类 `pipe`** 并完成 **权重加载与 pipeline 指针绑定**。
- 本文说明 **`pipe.infer` 触发后**，上述 **`Pi05Pipeline.forward` / Thor 的 `*_forward` 函数** 如何 **按阶段、按层** 执行。
- 调试时可结合 **`TLLM`/FlashRT 日志** 与 **Nsight** 确认：瓶颈常在 **大块 FP8 GEMM** 与 **FA2**，其次为 **每层融合 launch 数量**。

---

## 8. 关键源文件索引

| 路径 | 内容 |
|------|------|
| `flash_rt/models/pi05/pipeline_rtx.py` | RTX **`Pi05Pipeline`**：vision/encoder/decoder 实现、`run_pipeline`、`calibrate_fp8`、`autotune_gemms`、`record_infer_graph`、`forward` |
| `flash_rt/models/pi05/pipeline_thor.py` | Thor **Pi0.5 `decoder_forward`**（及 calibrate） |
| `flash_rt/hardware/thor/shared_primitives.py` | Thor **`siglip_forward` / `postln_project` / `encoder_forward`** |
| `flash_rt/hardware/thor/attn_backend.py` | **`ThorFlashAttnBackend`**、`make_pi05_attention_spec` |
| `flash_rt/hardware/rtx/attn_backend.py` | **`AttnBackend` 协议**、**`RtxFlashAttnBackend`**、FA2 代理与环境变量说明 |
| `flash_rt/frontends/torch/pi05_thor.py` / `pi05_rtx.py` | Torch 侧 **如何构造 pipeline、写缓冲、调用 infer**（与 `load_model` 返回的 `VLAModel` 直连） |

---

*文档版本与 FlashRT 源码同步；若接口更名请以仓库为准。*
