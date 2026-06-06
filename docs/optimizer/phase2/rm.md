# Phase 2 Roadmap：ViT / LLM Prefix 优化空间

> 背景：当前 `model_optimizer` 已经用 FlashRT denoise 接管 Pi0.5 denoise 热路径，denoise 侧大部分高收益融合点已经完成；ViT 也已经支持多视角 image batch。下一阶段优化重点应从 denoise kernel 转向 **prefix 阶段**：图像/语言 prefix 构建、PaliGemma LLM prefill、TRT KV cache 到 FlashRT decoder 的传递链路。

---

## 1. 当前推理主路径

典型混合部署路径：

```text
Policy.infer(obs)
  -> _preprocess_observation
  -> embed_prefix(images, lang)
       - ViT / SigLIP image features
       - language embedding
       - prefix_embs / pad_masks / att_masks 拼接
  -> prefix LLM forward(use_cache=True)
       - PaliGemma language model
       - 输出 past_key_values
  -> KV adapter
       - DynamicCache -> stack
       - valid_prefix trim
       - key pair-interleave
       - copy 到 FlashRT Kc/Vc slab
  -> FlashRT denoise full loop
```

对应关键代码：

| 阶段 | 文件 | 说明 |
|------|------|------|
| FlashRT hybrid 主循环 | `src/model_optimizer/infer/native/pi05_executor.py` | `sample_actions_flashrt` |
| Prefix LLM 导出 | `src/model_optimizer/models/pi05/llm.py` | LLM ONNX/TRT 子图 |
| TRT hook | `src/model_optimizer/infer/tensorrt/pi05_trt_engine_setup.py` | ViT / LLM / embed_prefix 安装 |
| ViT 子图 | `src/model_optimizer/models/pi05/vit.py` | SigLIP + projector |
| embed_prefix 子图 | `src/model_optimizer/models/pi05/embed_prefix.py` | image + language prefix |
| FlashRT decoder | `src/model_optimizer/infer/native/flashrt_decoder/` | denoise full-loop |
| Profiling | `src/model_optimizer/infer/perf/stage_perf.py` | `embed_prefix` / `prefix_llm` / `denoise.total` |

---

## 2. 优化判断原则

denoise 已经优化后，不应继续凭直觉优化单个 kernel。下一步先用阶段耗时确认瓶颈：

```text
prefix_llm / total
embed_prefix / total
KV_adapter / total
host_preprocess / total
denoise.total / total
```

建议默认打开：

```bash
export PI05_PI0_PROFILE=1
export MO_TRT_HOOK_STATS=1
```

重点看：

| 指标 | 含义 |
|------|------|
| `embed_prefix` | ViT + language embedding + prefix 拼接 |
| `prefix_llm` | PaliGemma LLM prefill / KV 生成 |
| `flashrt.setup` | FlashRT buffer / prompt setup |
| `denoise.total` | FlashRT full denoise loop |
| `trt.llm.forward` | TRT LLM 子图耗时 |
| `trt.vit.get_image_features` | TRT ViT 子图耗时 |

如果 `denoise.total` 已明显低于 prefix 阶段，则 Phase 2 优先级应转向 LLM / ViT / host 数据链路。

---

## 3. 最高优先级：Prefix LLM

### 3.1 LLM TRT CUDA Graph / 固定 shape

现状：

- 配置 `llm_engine` 后，PaliGemma language model forward 可被 TRT engine 替换。
- FlashRT hybrid 中 `sample_actions_flashrt` 仍调用 `paligemma_with_expert.forward(..., use_cache=True)`，已 hook 的 language model 会命中 TRT。
- 但 prefix 阶段仍可能有动态 shape、dtype cast、contiguous、output clone 等开销。

优化方向：

1. 固定 `enc_seq` / prompt length / camera count，让 LLM engine 尽量命中固定 profile。
2. 确认 `trt_cuda_graph=true` 在 FlashRT hybrid 下没有被关闭。
3. 检查 `Engine._prepare_input_tensor` 是否每帧触发 dtype cast / `.cuda()` / `.contiguous()`。
4. 检查 TRT 输出是否有不必要的 `last_hidden_state` 搬运，prefix denoise 实际主要消费 KV。

预期收益：**中到高**。denoise 被压缩后，prefix LLM prefill 往往会成为端到端主耗时之一。

工作量：**中等**。主要是配置验证、engine profile 固定、IO binding / graph capture 稳定性。

### 3.2 LLM 输出裁剪：只保留 KV

现状：

prefix 阶段真正传给 FlashRT denoise 的是 `past_key_values`。如果 TRT LLM engine 同时输出 `last_hidden_state` 且后续没有使用，可能有多余输出绑定和拷贝。

优化方向：

- 导出/构建 LLM engine 时支持 KV-only 输出模式。
- Python hook 里避免构造不必要的 `BaseModelOutputWithPast.last_hidden_state` 或避免其 device copy。

预期收益：**中等**，取决于 engine 输出绑定和实际拷贝占比。

工作量：**中等**。需要确认 downstream 是否完全不依赖 `last_hidden_state`。

### 3.3 Prefix mask / position_ids 缓存

现状：

`sample_actions_flashrt` 每帧构造：

```python
prefix_cumsum
prefix_att_2d_masks
prefix_pad_2d_masks
prefix_position_ids
prefix_att_2d_masks_4d
```

优化方向：

- 固定 prompt/camera layout 时缓存 mask / position_ids 模板。
- 若 state token 长度固定，可只更新有效长度或直接复用。
- 将 bool 2D mask 到 TRT 需要的 4D additive mask 预先生成。

预期收益：**低到中**，但实现风险低。

---

## 4. 高优先级：KV 传递和 Layout 适配

### 4.1 当前 KV 传递链路

当前 FlashRT hybrid 中：

```text
TRT/PyTorch LLM -> DynamicCache
  -> _stack_past_key_values()
  -> valid_prefix trim
  -> optional even padding
  -> FlashRtDecoderBackend.run()
  -> fill_prefix_kv_from_trt()
  -> K pair-interleave + copy_ 到 Kc/Vc
```

关键代码在 `src/model_optimizer/infer/native/pi05_executor.py`：

```python
past_keys, past_values = self._stack_past_key_values(past_key_values)
valid_prefix = prefix_pad_masks[0].to(dtype=torch.bool)
enc_seq = int(valid_prefix.sum().item())
past_keys = past_keys[:, :, valid_prefix, :].contiguous()
past_values = past_values[:, :, valid_prefix, :].contiguous()
```

以及 `flashrt_decoder/driver.py::fill_prefix_kv_from_trt`：

```text
TRT KV [L, 1, enc_seq, HD]
  -> K pair-interleave
  -> Kc[:, :enc_seq, :]
  -> Vc[:, :enc_seq, :]
```

### 4.2 优化方向：KV 零拷贝 / 少拷贝

潜在优化：

1. **LLM engine 直接输出 FlashRT 期望 K layout**
   - 让 key 在 TRT 图内完成 pair-interleave。
   - Python 侧不再做 reshape/permute/contiguous。

2. **LLM engine 直接输出到 FlashRT prefix slab**
   - 理想路径是 TRT output binding 指到 `Kc/Vc` prefix 区。
   - 避免 DynamicCache / stack / copy_ 中间链路。

3. **减少 valid_prefix trim**
   - 构建/输入阶段让 prefix physical layout 直接等于有效 token。
   - 或固定 padding 策略，避免每帧 bool indexing。

4. **偶数 enc_seq 固化**
   - 当前奇数时 duplicate 最后一个 token 以满足 Thor kernel 假设。
   - 可以在 prompt 构建或 engine profile 层保证 `enc_seq` 恒偶。

预期收益：**中到高**。当 denoise 变快后，每帧 KV copy/layout transform 会变得更显眼。

工作量：**中到高**。TRT output binding 到外部 slab 需要仔细处理生命周期、shape、stream 和 graph capture。

---

## 5. ViT / embed_prefix 剩余空间

ViT 已做 image batch 后，继续抠 ViT kernel 本体的 ROI 降低。剩余重点是 **embed_prefix 整体融合、前后处理、数据拷贝**。

### 5.1 确认 ViT batch views 已启用

若仍使用 TRT ViT 但没有 `vit_batch_views`，多视角仍会逐 view 调用 `get_image_features`。

检查项：

- 配置里 `vit_batch_views=true`。
- `vit_engine` 的 dynamic batch profile 覆盖视角数 `V`。
- `trt.vit.get_image_features` 调用次数应从每帧 `V` 次降为 1 次。

预期收益：**高**，但若已经启用则无额外空间。

### 5.2 embed_prefix 整图 TRT

现状：

即使 ViT 多视角 batching，运行时仍可能保留：

```text
image feature list
language embedding
prefix_embs concat
pad_masks / att_masks concat
```

优化方向：

- 使用 `embed_prefix.engine`：image + language + concat 整段 TRT。
- 或将 `models/pi05/embed_prefix.py` 中的 batched prefix 逻辑迁移到 runtime hook。
- 避免 Python list 拼接和小 torch op。

预期收益：**中等**。当 ViT 本体已优化后，prefix 拼接和小 op 占比会上升。

工作量：**中等**。需要确认数值对齐和 mask 输出格式。

### 5.3 ViT TRT CUDA Graph / dtype 对齐

检查项：

- `trt_cuda_graph=true` 是否对 ViT engine 生效。
- 输入 dtype 是否与 engine 期望一致，避免每帧 cast。
- 输入是否已经 contiguous，避免 `Engine` 内部重排。
- 输出是否有不必要 `clone()`。
- `trt_vit_scale_fix` 是否只在需要时开启，避免重复乘 `sqrt(hidden_size)`。

预期收益：**低到中**，但容易验证。

### 5.4 图像预处理和 host 数据链路

现状风险：

```text
numpy image
  -> torch.from_numpy
  -> CPU resize / normalize
  -> cuda copy
  -> CHW/HWC 转换
```

优化方向：

1. 固定输入尺寸，避免 `_preprocess_observation` 二次 resize。
2. 使用 pinned memory + `non_blocking=True`。
3. 将 resize / normalize / layout transform 放到 GPU fused preprocess。
4. 如果使用 FlashRT SigLIP，消除 GPU -> CPU -> GPU 回环。

预期收益：**中等**，尤其 host overhead 高时。

工作量：从低到高不等。pinned/non_blocking 简单，GPU fused preprocess 较重。

---

## 6. 可选结构性优化

### 6.1 Prefix 部分缓存

Pi0.5 的 state 进入 prompt，image 每帧变化，因此 prefix 不能整体缓存。但仍可拆：

| 可缓存项 | 说明 |
|----------|------|
| 固定文本 prompt embedding | 如果任务文本固定 |
| mask / position 模板 | shape 和 padding 固定时 |
| tokenizer 输出 | 任务文本固定时 |
| 相机 layout | 视角数固定时 |

收益：**低到中**，但能减少 Python overhead。

### 6.2 Prefix + denoise 端到端 CUDA Graph

理想路径：

```text
embed_prefix TRT graph
  -> prefix_llm TRT graph
  -> KV adapter
  -> FlashRT denoise graph
```

难点：

- TRT IO binding 和外部 FlashRT slab 的地址稳定性。
- prefix length / valid mask 动态。
- 图像输入地址和 dtype/shape 固定。
- DynamicCache Python 对象不能在 graph 内。

收益：**中到高**，但工程复杂度高。建议在 KV adapter 和 fixed-shape TRT graph 稳定后再做。

### 6.3 Batch > 1

当前 FlashRT hybrid 明确限制 batch=1。若未来评测/仿真多 env 并行，支持 B>1 可能显著提升吞吐，但会扩大：

- ViT batch 维语义
- LLM KV layout
- FlashRT decoder buffer layout
- denoise kernel grid

这是吞吐优化，不一定改善单帧 latency。

---

## 7. 不建议优先做的方向

| 方向 | 原因 |
|------|------|
| 继续微调 FlashRT denoise kernel | 主要高收益点已经完成；除非 profiler 显示 denoise 仍是 top-1 |
| 单独替换 Pi0.5 Thor decoder attention 为 `attn.run` | Thor 上最终仍是 `attention_qkv_fp16`，只是多一层 backend wrapper |
| 继续抠 ViT kernel 本体 | image batch 后边际收益下降；先看 prefix_llm / KV / preprocess |
| 在 C5 GEMM 后直接输出 fp8 | quant 点错误；GeGLU 非线性应在 fp16/fp32 域做，C6 后再 quant |
| 重开 torch.compile | 与 TRT / FlashRT hook 路径容易冲突，当前已有 eager restore 逻辑 |

---

## 8. 推荐迭代计划

### Milestone A：建立可靠 baseline

- 开启 `PI05_PI0_PROFILE=1`、`MO_TRT_HOOK_STATS=1`。
- 记录至少 20 次 post-warmup 推理。
- 汇总：
  - `sample_actions`
  - `_preprocess_observation`
  - `embed_prefix`
  - `prefix_llm`
  - `flashrt.setup`
  - `denoise.total`
  - `trt.vit.get_image_features`
  - `trt.llm.forward`

产出：一张阶段占比表，作为 Phase 2 优先级依据。

### Milestone B：LLM 固定 shape + CUDA Graph

- 确认 `llm_engine` 已启用。
- 固定 prompt length / camera count / engine profile。
- 验证 `trt_cuda_graph=true` 对 LLM 可用。
- 检查输入 dtype / contiguous / clone。

成功标准：

- `trt.llm.forward` latency 稳定下降。
- 没有数值偏差或 graph capture fallback。

### Milestone C：KV adapter 减拷贝

- 单独计时 `_stack_past_key_values`、valid trim、K interleave、Kc/Vc copy。
- 先做低风险优化：固定 even enc_seq、减少 bool indexing、避免不必要 `.contiguous()`。
- 再评估 TRT 直接输出 FlashRT layout。

成功标准：

- `prefix_llm` 后到 `denoise.total` 前的非模型开销下降。
- FlashRT denoise 输入 KV 数值保持一致。

### Milestone D：embed_prefix 整图化

- 对比三种路径：
  1. TRT ViT + `vit_batch_views`
  2. `embed_prefix.engine`
  3. FlashRT SigLIP + TRT LLM
- 根据 `embed_prefix` 占比决定是否推进整图 TRT 或 FlashRT SigLIP 路径。

成功标准：

- `embed_prefix` latency 下降。
- prefix KV 与原路径数值在可接受误差内。

### Milestone E：Host preprocess

- 检查是否存在重复 resize / dtype conversion / CPU-GPU 同步。
- 引入 pinned memory + non_blocking copy。
- 如 host 占比仍高，再考虑 GPU fused preprocess。

---

## 9. 当前推荐优先级

| 优先级 | 任务 | 预期收益 | 工作量 |
|--------|------|----------|--------|
| P0 | 建立 prefix/denoise 分阶段 baseline | 必需 | 低 |
| P1 | LLM TRT CUDA Graph + fixed shape | 中到高 | 中 |
| P1 | KV adapter 减拷贝 / layout 对齐 | 中到高 | 中到高 |
| P2 | embed_prefix 整图 TRT 或 runtime batched prefix | 中 | 中 |
| P2 | ViT TRT graph / dtype / clone 检查 | 低到中 | 低 |
| P2 | Host preprocess pinned/non_blocking | 低到中 | 低 |
| P3 | Prefix + denoise 端到端 CUDA Graph | 中到高 | 高 |
| P3 | Batch > 1 | 吞吐收益 | 高 |

---

## 10. 简短结论

Phase 2 不应继续围绕 denoise 做主线优化。当前更可能有收益的区域是：

1. **Prefix LLM**：TRT graph、fixed shape、KV-only 输出。
2. **KV 传递**：减少 `DynamicCache -> stack -> trim -> interleave -> copy`。
3. **embed_prefix**：从「ViT batch」推进到「整段 prefix 图化」。
4. **host preprocess**：减少 CPU resize、拷贝、layout 转换。

经验判断：如果当前 denoise 已经明显下降，**LLM prefix + KV adapter** 的收益概率高于继续优化 ViT kernel 本体。
