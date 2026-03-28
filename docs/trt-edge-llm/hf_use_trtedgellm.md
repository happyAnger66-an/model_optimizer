# Hugging Face 模型到 TensorRT Edge LLM（EdgeLLM 包装）流程说明

本文档说明 `third_party/TensorRT-Edge-LLM` 中，如何将 Hugging Face（HF）因果语言模型加载并 **包装为 EdgeLLM**，再 **导出 ONNX** 的总体流程与原理。核心入口为 `tensorrt_edgellm/onnx_export/llm_export.py` 中的 `export_llm_model`。

---

## 1. 目标与角色分工

| 阶段 | 作用 |
|------|------|
| **HF 加载** | 使用 `AutoModelForCausalLM` / `AutoModelForImageTextToText`（及 Phi4MM 等特殊路径）读出原始权重与结构。 |
| **EdgeLLM 包装** | 构造 `EdgeLLMModelForCausalLM`：保留 `embed_tokens`、`norm`、`lm_head`（可选词表裁剪），将 **每一层 Transformer block 替换为** `EdgeLLMDecoderLayer`，以匹配 Edge 推理栈的 KV/RoPE/注意力插件接口。 |
| **ONNX 导出** | 用与运行时一致的 **dummy 输入** 做 `torch.onnx.export`（经 `export_onnx`），并注册 **Attention / GatherND / INT4 GEMM** 等自定义 ONNX symbolic，便于后续 TensorRT 构建。 |
| **产物落盘** | `config.json`（经 `export_llm_config`）、tokenizer、可选 processor、`processed_chat_template.json`、可选 `vocab_map.safetensors`。 |

---

## 2. `export_llm_model` 流程（自上而下）

源码位置：`tensorrt_edgellm/onnx_export/llm_export.py` 中 `export_llm_model`。

```
export_llm_model(model_dir, output_dir, device, is_eagle_base, ...)
    │
    ├─ 可选：load_reduced_vocab_map → reduced_vocab_size, vocab_map
    │
    ├─ load_llm_model(...)     ← HF + EdgeLLM 包装（见下节）
    │
    ├─ replace_torch_quant_linear_with_int4_plugin(model)
    │     （若为 GPTQ：TorchQuantLinear → Int4GemmPluginModule，并注册 INT4 symbolic）
    │
    ├─ create_dummy_inputs(...)   ← 与 ONNX I/O 形状、命名对齐的占位张量
    │
    ├─ export_model_to_onnx(...)  ← 组 inputs tuple、input_names、dynamic_axes、注册插件 symbolic → export_onnx
    │
    ├─ export_llm_config → 写入 output_dir/config.json
    │
    ├─ tokenizer.save_pretrained / processor.save_pretrained
    │
    └─ chat template：validate + copy 或 process_chat_template
       可选：复制 vocab_map.safetensors
```

**要点：**

- `is_eagle_base=True` 时导出的是 EAGLE3 **基座**（与标准 LLM 共用同一套导出框架，但 dummy 与输出名含 `hidden_states`）。
- **Draft 模型**走独立入口 `export_draft_model`（`load_eagle3_draft_model` + `Eagle3DraftModel`），不在 `export_llm_model` 内完成。

---

## 3. HF → EdgeLLM：包装发生在哪里？

包装逻辑在 **`tensorrt_edgellm/llm_models/model_utils.py`** 的 `load_llm_model`：

1. **`load_hf_model`**  
   - 多数情况：`AutoModelForCausalLM.from_pretrained(..., _attn_implementation="eager")`。  
   - 失败则尝试 `AutoModelForImageTextToText`（多模态）。  
   - Phi4MM 等走本地 `modeling_phi4mm` 的动态导入 workaround。  
   - 非 GPTQ 时整体 `.to(torch.float16)`（与 `dtype='fp16'` 一致）。

2. **`is_vlm(model_dir)`**  
   - 根据 config 判断是否视为 VLM，决定 **`use_prompt_tuning`**（影响嵌入路径与 ONNX 是否带 `image_embeds`）。

3. **`set_dynamic_quant(model, dtype)`**  
   - 对 NVFP4 / MXFP8 等 ModelOpt 量化层设置 ONNX/TRT 侧 quantizer 元数据，便于导出与推理对齐。

4. **构造 Edge 模型**  
   ```text
   edge_model = EdgeLLMModelForCausalLM(
       hf_model,
       is_eagle_base,
       use_prompt_tuning,
       reduced_vocab_size,
       vocab_map,
   )
   del hf_model  # 释放原始 Module，减少显存
   ```

5. 返回 **`(edge_model, use_prompt_tuning, tokenizer, processor)`**，供 `export_llm_model` 后续使用。

因此：**「包一层 EdgeLLM」是在 `load_llm_model` 里完成的**；`export_llm_model` 拿到的是已经包装好的 `EdgeLLMModelForCausalLM`。

---

## 4. 包装原理（`EdgeLLMModelForCausalLM`）

实现见 **`tensorrt_edgellm/llm_models/models/llm_model.py`**。

### 4.1 从 HF 拆出「语言塔」

- **普通 Causal LM**：`language_model = hf_model.model`，`config = hf_model.config`。  
- **`use_prompt_tuning`（VLM 等）**：若有 `hf_model.language_model` 则用其 + `text_config`；否则（如 Phi4MM）用 `hf_model.model` 与主 config。

### 4.2 主干 `EdgeLLMModel`

- **保留并 cast**：`embed_tokens`、`norm`（与 HF 中 language tower 一致）。  
- **替换 Decoder**：`hf_model.layers` 中每一层传入 **`EdgeLLMDecoderLayer(hf_layer, ...)`**，在内部把标准 attention/MLP 换成 **带 Edge 注意力插件契约** 的实现。  
- **Forward 契约**：显式传入 `past_key_values`、`rope_rotary_cos_sin`、`context_lengths`、`kvcache_start_index` 等，与 HF 默认 `forward(input_ids=..., past_key_values=...)` 不同，目的是与 **固定 KV cache 布局 + 外部 RoPE** 的推理引擎一致。

### 4.3 Causal LM 头

- 默认复用 **`hf_model.lm_head`**。  
- 若提供 `reduced_vocab_size` 与 `vocab_map`，通过 **`reduce_lm_head`** 做词表子集映射，减小输出维度。  
- 前向末尾：对最后一层 hidden 用 **`custom_gather_nd`** 按 `last_token_ids` 取位置，再过 `lm_head` 得 **logits**（float32）。  
- **`is_eagle_base`**：除 logits 外还拼接多层 hidden 供 EAGLE3 使用（与 ONNX 多输出 `hidden_states` 对应）。

---

## 5. ONNX 导出阶段的输入组织（`export_model_to_onnx`）

`create_dummy_inputs` 按 **层数、hidden、head_dim、rotary_dim、max_position_embeddings** 等构造：

- 每层 **`past_key_values.{i}`**：形状约定为 `(batch, 2, num_kv_heads, seq_or_past_len, head_dim)`，FP16。  
- **`rope_rotary_cos_sin`**、**`context_lengths`**、**`last_token_ids`**、**`kvcache_start_index`**、**`input_ids`**。  
- 标准模型：`position_ids` / `attention_mask` 在导出时传 **`None`**（占位），命名上不暴露给 ONNX；**EAGLE** 则传入并命名为 `attention_pos_id`、`attention_mask`。  
- VLM：**`image_embeds`**；Qwen3-VL text：**`deepstack_visual_embeds`** → ONNX 上命名为 `deepstack_features.{i}`。

**动态轴**（`dynamic_axes`）为 `batch_size`、`seq_len`、`past_len`、`present_kv_cache_len` 等预留符号名，便于 TRT profile。

**Symbolic 注册**（导出前）：

- `register_attention_plugin_onnx_symbolic_functions`  
- `register_gather_nd_onnx_symbolic_functions`  
- GPTQ 路径下 earlier：`register_int4_gemm_plugin_onnx_symbolic_functions`

最终调用 **`export_onnx(model, inputs, ...)`** 写出单文件 ONNX（及可能的图外科尔）。

---

## 6. 与本文档相关的源码索引

| 文件 | 说明 |
|------|------|
| `onnx_export/llm_export.py` | `export_llm_model`、`create_dummy_inputs`、`export_model_to_onnx` |
| `llm_models/model_utils.py` | `load_llm_model`、`load_hf_model`、`load_eagle3_draft_model` |
| `llm_models/models/llm_model.py` | `EdgeLLMModel`、`EdgeLLMModelForCausalLM` 包装与 forward |
| `onnx_export/config_export.py` | `export_llm_config` |
| `onnx_export/onnx_utils.py` | `export_onnx` 及 NVFP4/图优化等后处理 |
| `model_optimizer/.../pi05/llm_with_trtedgellm.py` | π0.5：在 `GemmaModel` 上仅包装 `EdgeLLMAttention`（`GemmaAttentionTrtEdge`） |

---

## 7. 小结

- **HF 模型**先按标准方式加载；**真正面向 Edge/TRT 的图**是 **`EdgeLLMModelForCausalLM`**：共享嵌入与 LM 头，**Decoder 全部替换为 Edge 定制层**，前向接口围绕 **KV cache、外部 RoPE、按 batch 收集最后 token**。  
- **`export_llm_model`** 负责：可选词表裁剪与 GPTQ 插件替换 → 造 dummy → 带插件 symbolic 导出 ONNX → 写 config/tokenizer/chat template。  
- 若要对照实现细节，建议同时打开 **`llm_export.py`（456 行起）** 与 **`model_utils.load_llm_model`**、**`llm_model.py`**。  
- π0.5 将 PaliGemma 的 **`GemmaModel`** 接到 Edge 的说明与示例见下文 **第 8 节**。

---

## 8. PI0.5 PaliGemma 语言塔对接（`llm_with_trtedgellm.py`）

TRT-Edge-LLM 的主要 ONNX/TRT 优化在 **`EdgeLLMAttention`**（``trt::attention_plugin``）。整塔 **`EdgeLLMModel`** 还会替换 **`EdgeLLMDecoderLayer`**，其 layernorm 约定与 OpenPI 的 **`GemmaRMSNorm` / gated residual** 不一致。  
本仓库采用 **仅替换 attention** 的策略：在 **`GemmaModel`** 各层把 **`self_attn`** 包成 **`GemmaAttentionTrtEdge`** —— **eager 前向仍走原生 `GemmaAttention`**；同权重的 **`EdgeLLMAttention`** 挂在 **`.edge`** 上，供按需导出插件子图或自定义整塔 trace。

### 8.1 设计要点

1. **为何不整塔替换 `EdgeLLMModel`**  
   **`EdgeLLMDecoderLayer`** 对 layernorm 是单入参、单返回值；**`GemmaDecoderLayer`** 需要 **`(hidden, gate)`** 与 **`GemmaRMSNorm`**。整塔 Edge 会与现有 PI0 / OpenPI 前向不兼容。

2. **做法（仅 attention）**  
   - **`install_gemma_edge_attention_wrappers(gemma_model)`**：就地令每层 **`self_attn` → `GemmaAttentionTrtEdge(native)`**，**`forward(...)` 仍委托 `native.forward`**，与 **`llm.LLM`**、联合注意力等逻辑数值一致。  
   - **`module.edge`**：即 **`EdgeLLMAttention`**，与 **`native`** 共享 **q/k/v/o_proj**；**`edge.forward(...)`** 使用 Edge 约定的 **KV / RoPE 张量**（见本文 **第 5 节** 与 `llm_export.py`）。  
   - **无二倍 Gemma 塔**：不再 **`deepcopy` 整塔 + `EdgeLLMModel`**；显存上仅多一层薄包装与 **`EdgeLLMAttention`** 对象（权重仍一份）。

3. **`GemmaAttentionTrtEdge` 与 OpenPI**  
   **`layer.self_attn.q_proj`** 等通过 **`__getattr__`** 转发到 **`native`**，兼容 **`gemma_pytorch`** 里直接访问 **`self_attn` 子模块** 的写法。

4. **`Pi05TrtEdgeLanguageModel` / `LLMWithTrtEdgeLLM`**  
   - 可选 **`wrap_edge_attention=True`**（默认）：初始化时对 **`hf_gemma`** 安装上述包装。  
   - **`LLMWithTrtEdgeLLM`** **不继承** **`LLM`**，独立实现与 PI0 一致的 **`forward` / `val` / `quantize`**，并混入 **`Model`**。  
   - **`export` / `export_onnx`**：追踪 **`GemmaModelEdgeOnnxExport`**（Gemma 结构 + 各层走 **`EdgeLLMAttention.forward`**），导出前调用 **`register_attention_plugin_onnx_symbolic_functions`**，使 ONNX 含 **`trt::attention_plugin`**。外壳输入仍为 **`inputs_embeds` / `attention_mask` / `position_ids`**（与 **`LLM.export`** 对齐）；KV/RoPE 等在 **`GemmaModelEdgeOnnxExport` 内部** 按 Edge dummy 约定构造。  
   - **`gemma_decoder()`**（旧名 **`get_edgellm_core()`**）：返回已包装 attention 的 **`GemmaModel`**。  
   - **`edge_attention_modules()`**：按层返回 **`EdgeLLMAttention`** 列表。

5. **限制与依赖**  
   - **`config.use_adarms=True`** 时 **`NotImplementedError`**。  
   - 需 **`third_party/TensorRT-Edge-LLM`** 在 **`PYTHONPATH`** 中（模块内会尝试自动插入）。  
   - PyTorch 下 **`EdgeLLMAttention.forward`** 内 **`attention_plugin`** 为 **dummy**，不要用 **`edge.forward`** 对齐 HF 数值；对齐仍用 **`native.forward`**。

### 8.2 使用示例

```python
from model_optimizer.models.pi05.llm_with_trtedgellm import (
    swap_pi05_language_model_to_trt_edgellm,
    LLMWithTrtEdgeLLM,
    install_gemma_edge_attention_wrappers,
    GemmaModelEdgeOnnxExport,
)

# 方式 A：set_decoder(bundle)，模块树 paligemma → bundle → Gemma（已包 attention）
swap_pi05_language_model_to_trt_edgellm(policy._model)

# 方式 B：与 llm.LLM 对称；就地安装 Edge 注意力包装；export 自动走 GemmaModelEdgeOnnxExport（含插件）
llm = LLMWithTrtEdgeLLM.construct_model(pi05_model)
gemma = llm.gemma_decoder()
edges = llm.edge_attention_modules()
llm.export(output_dir)  # llm.onnx 内含 trt::attention_plugin

# 也可只对已有 GemmaModel 安装包装：
install_gemma_edge_attention_wrappers(paligemma.get_decoder())
export_net = GemmaModelEdgeOnnxExport(paligemma.get_decoder())  # 需已 install_gemma_edge_attention_wrappers
```
