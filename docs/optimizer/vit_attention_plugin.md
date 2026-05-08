# ViTAttentionPlugin（TensorRT-Edge-LLM）

本文总结 NVIDIA TensorRT-Edge-LLM 中 **`ViTAttentionPlugin`** 的实现、用法、设计动机与主要优化。源码位于本仓库子模块 **`third_party/TensorRT-Edge-LLM`**（若你单独克隆 Edge-LLM，则对应其 `cpp/plugins/vitAttentionPlugin/` 与 `tensorrt_edgellm/` 下相关文件）。

---

## 1. 实现要点（C++ 插件）

- **接口类型**：`nvinfer1::IPluginV2DynamicExt`，通过 `REGISTER_TENSORRT_PLUGIN` 注册；引擎推理时需加载 Edge-LLM 插件动态库，否则无法解析图中的自定义节点。
- **插件名 / 版本**：`ViTAttentionPlugin`，版本字符串 `"1"`（见 `vitAttentionPlugin.cpp` 中常量）。

### 1.1 输入与输出

| 索引 | 含义 | 类型与形状约定 |
|------|------|----------------|
| 0 | Q | `FP16`，`kLINEAR`，三维 **`[total_S, num_heads, head_size]`** |
| 1 | K | 同 Q |
| 2 | V | 同 Q |
| 3 | `cu_seqlens` | `int32`，一维，长度 **`B + 1`**：将 `B` 个样本在序列维拼成 `total_S` 时的前缀和（与常见 FlashAttention 风格 API 一致） |
| 4 | `max_seqlen_carrier` | `int32` 一维张量；**仅使用其第 0 维长度**作为运行时的 `max_seqlen` 提示，**元素取值可忽略** |
| 输出 0 | attention 输出 | 与 Q 相同形状 **`[total_S, H, D]`** |

注释中强调：**Python/ONNX 导出必须保证 Q/K/V 为上述 head-major 三维布局**（见 `vitAttentionPlugin.cpp` 中 `supportsFormatCombination` 的说明）。

### 1.2 `enqueue` 中的实际计算

两条执行路径（编译与硬件能力决定）：

1. **CuTe DSL ViT FMHA**（定义了 `CUTE_DSL_FMHA_ENABLED` 且 `CuteDslFMHARunner::canImplementViT` 为真且内核模块加载成功时）：调用 `CuteDslFMHARunner::run`。
2. **否则**：使用 **`ContextFMHARunner`**，`AttentionInputLayout::SEPARATE_Q_K_V`，`ContextAttentionMaskType::PADDING`；`cu_q_seqlens` 与 `cu_kv_seqlens` 指向同一块 `cu_seqlens` 设备内存，最后 `dispatchFMHAKernel`。

构造阶段会按当前 GPU **SM 版本**与 **head_size** 判断是否能实现；若不能实现则抛错，避免生成无法执行的引擎配置。

### 1.3 其它接口行为

- **`getWorkspaceSize`**：当前实现返回 **0**（工作区由底层 runner 自管或不需要额外 workspace）。
- **序列化字段**：`num_heads`、`head_size`（见 `getSerializationSize` / `serialize`）。

---

## 2. 用法（Python → ONNX → TensorRT）

### 2.1 Torch 侧自定义算子

- 算子命名空间：**`trt::vit_attention_plugin`**（`tensorrt_edgellm/llm_models/layers/attention_plugin.py`）。
- 前向实现为 **`torch.zeros_like(q)`**，用于在 PyTorch 中占位；**真实数学在 TRT 插件的 CUDA 核中执行**，训练/验证若依赖正确 attention，需使用未 patch 的原始模块或自行替换实现。

### 2.2 ONNX 导出

- 通过 **`register_custom_op_symbolic`** 将 `trt::vit_attention_plugin` 映射为 ONNX 中的 **`trt::ViTAttentionPlugin`**，并注册对应的 **`OpSchema`**（`vit_attention_plugin_schema`），属性包含 **`num_heads`**、**`head_size`**，与 C++ `ViTAttentionPluginCreator` 的字段一致。

### 2.3 视觉模型中的典型接法

以 Qwen2-VL 的 patch 为例（`tensorrt_edgellm/visual_models/qwen2_vl_model.py`）：

- 在 **`apply_rotary_pos_emb_vision`** 之后将 Q/K/V 转为 **`float16`**，形状为 **`[total_S, H, D]`**；
- 调用 **`vit_attention_plugin(..., cu_seqlens, max_seqlen_carrier, num_heads, head_size)`**；
- 输出再 **`reshape`** 为 **`[total_S, H * D]`** 后接 **`proj`**。

Qwen2.5-VL、Qwen3-VL 等视觉模型中有同类用法（同一 `attention_plugin` 模块）。

### 2.4 运行时依赖

加载含该节点的 TensorRT 引擎时，必须保证 **Edge-LLM 插件共享库**在进程内可用（并正确设置插件搜索路径 / 预加载），否则解析或执行 `ViTAttentionPlugin` 会失败。

---

## 3. 为什么要单独做这样一个插件？

| 动机 | 说明 |
|------|------|
| **变长序列 / 拼接 batch** | ViT 在多图、多分辨率等场景下常用 **`total_S` 拼接 + `cu_seqlens`** 描述子序列；用专用 FMHA 可在一次内核调度中处理，避免固定 `(B, S_max, …)` 上整 pad 矩阵的浪费。 |
| **与 LLM 的 AttentionPlugin 分离** | ViT 版为 **独立 Q/K/V、无 KV cache、插件内不含 RoPE**（RoPE 在图上前置完成）。LLM 的 `AttentionPlugin` 需处理 cache、RoPE、树注意力等，接口更重，不宜混用。 |
| **相对 TRT 原生子图展开** | 将 attention 拆成多个 MatMul/Softmax 等算子往往 **kernel 碎、访存差**；统一到项目内 **context FMHA** 路径，与现有 kernel 体系一致，便于按 SM 维护与优化。 |

---

## 4. 优化与收益（及当前限制）

### 4.1 主要优化方向

- **融合 FMHA**：`ContextFMHARunner` 或 CuTe DSL 路径减少中间张量与 launch 次数，提高带宽与计算利用率。
- **变长语义**：`cu_seqlens` + padding 类型 attention，按真实子序列计算，减少无效 token 上的算力。
- **按 SM 选核**：构造时 `canImplement` 与 `loadContextFMHAKernels` / `loadViTKernelModule`，在目标架构上加载合适实现。
- **新 GPU 可选 CuTe DSL**：支持时走 `CuteDslFMHARunner`，否则回退 FMHA v2，兼顾新架构与兼容路径。
- **`max_seqlen_carrier`**：用张量 **shape** 传递 `max_seqlen`，便于动态维度下仍为底层 runner 提供明确上限，而不依赖 carrier 张量内的数值。

### 4.2 限制（实现与 schema 当前约定）

- **数据类型**：插件侧 **`FP16`（kHALF）**；导出图需与此一致。
- **布局**：Q/K/V 与输出均为 **`[total_S, num_heads, head_size]`** 线性格式；与部分框架默认的 `(B, S, H, D)` 需通过 reshape/permute 对齐。

---

## 5. 相关文件索引（Edge-LLM 子模块内）

| 角色 | 路径（相对于 `third_party/TensorRT-Edge-LLM`） |
|------|-----------------------------------------------|
| C++ 插件实现 | `cpp/plugins/vitAttentionPlugin/vitAttentionPlugin.{h,cpp}` |
| ONNX schema / 占位算子 / symbolic | `tensorrt_edgellm/llm_models/layers/attention_plugin.py` |
| 视觉模型中的调用示例 | `tensorrt_edgellm/visual_models/qwen2_vl_model.py`、`qwen2_5_vl_model.py`、`qwen3_vl_model.py` 等 |

**一句话**：`ViTAttentionPlugin` 为视觉 Transformer 中 **变长序列上的多头注意力** 提供 TensorRT 自定义节点，在引擎中落到 **融合多头注意力（FMHA）** 实现，以在边端 GPU 上相对「纯 ONNX 算子链」更易获得 **延迟与吞吐** 收益，并与 Qwen2/2.5/3-VL 等导出路径对齐。
