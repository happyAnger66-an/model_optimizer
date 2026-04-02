# Q&A：PI0.5 LLM KV Cache 导出/编译/TRT 数值问题定位总结

本文记录一次完整的排查过程：PI0.5（Gemma/Paligemma 语言塔）在 ONNX 导出与 TensorRT engine 运行时出现
**后续层 KV cache 全 0**、**输出塌缩为 0**、以及 **NaN 扩散** 等问题。目标是把定位路径固化下来，便于复用。

---

## 背景与现象

### 现象 A：ONNX/TRT 输出的 KV cache 只有第 0 层正常，后续层为 0

- PyTorch eager 下 `DynamicCache` 每层 KV 都有值。
- ONNX 导出后构建的 TRT engine 输出：
  - `present_key_values.0` 非 0
  - `present_key_values.1` 开始明显衰减，甚至全 0

### 现象 B：TRT 输出 `last_hidden_state` 全 0（或后半段变 NaN）

典型表现：
- `last_hidden_state abs_sum=0`（全 0），或
- `present_key_values.1` 的后半段 token 行开始出现 `nan`

---

## 排查过程（按时间线）

### 1) 怀疑 KV cache 导出逻辑：DynamicCache 在 Dynamo/ONNX 下不稳定

#### 初始实现（问题版本）
在 `src/model_optimizer/models/pi05/llm.py` 的 wrapper forward 中：

- `prefix_output = model(..., use_cache=True)`
- `past_key_values = prefix_output.past_key_values  # DynamicCache`
- 通过 Python 循环 `past_key_values[i]` 抽每层 KV，然后 `torch.cat`。

#### 结论
`DynamicCache.update()` 属于 **Python 对象内部状态的副作用更新**，`torch.onnx.export(dynamo=True)` 更偏向捕获
**纯张量数据流**；对这种“对象内 list 被更新”的场景，导出不稳定，容易出现：

- 图中只捕获第 0 层索引/更新
- 其它层在导出执行里变成默认占位（全 0）

#### 修复方向
实现 **导出专用 forward**，完全绕开 `DynamicCache.update`，将每层 K/V 作为“显式张量”产出：

- 输出改为 `present_key_values.{i}` per-layer（参考 TensorRT-Edge-LLM 的做法）
- 每层输出形状约定为 `[B, 2, Hkv, S, D]`（0=K，1=V）

相关文件：
- `src/model_optimizer/models/pi05/llm.py`
- `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/onnx_export/llm_export.py`
- `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/llm_models/models/llm_model.py`

---

### 2) 构建/运行 TRT engine 后仍异常：用 Engine dump 脚本定位“engine 本身算错”还是“上层拼接错”

为了把问题从“上层 wrapper/拼接”剥离出来，新增了一个独立脚本直接跑 `.engine`：

- `scripts/debug/dump_trt_engine_io.py`

脚本能力：
- 读取 engine 的输入/输出签名
- 构造 dummy 输入（含可配置的 `attention_mask`）
- 打印每个输入/输出的统计：
  - `min/max/mean/abs_sum`
  - `nan_count/inf_count`
  - `first_nan_seq_index`（对 `[B,S,...]` 或 `[B,2,H,S,D]` 定位 NaN 从哪个 token 行开始）

这一步的价值：
- 若 engine 输出本身就异常（全 0 / NaN），说明问题在 **ONNX 图/构建精度/数值路径**
- 若 engine 输出正常但上层异常，才继续查 executor/wrap

---

### 3) 关键根因 1：构建精度与 ONNX dtype 不一致（bf16 vs fp16）

现象复盘：
- engine 输出 dtype 是 bf16，但 build 时误指定 `--precision fp16`
- 表现为：
  - 第 0 层 KV 可能还“看起来正常”
  - 深层开始出现塌缩（0/NaN）

修复：
把 build precision 改成 **bf16** 后，深层 0 问题消失（或明显缓解）。

为避免再次踩坑，新增了 **精度一致性检查**：
- 推断 ONNX 图中主要浮点 dtype（bf16/fp16/fp32/mixed/unknown）
- 若 ONNX=bf16 但 build_cfg.precision=fp16（或相反），直接报错拦截

相关实现：
- `src/model_optimizer/trt_build/build.py`
  - `infer_onnx_float_precision()`
  - `validate_precision_matches_onnx()`
- `src/model_optimizer/trt_build/cli.py`
  - build_cli 调用 `validate_precision_matches_onnx`
- `src/model_optimizer/webui/runners/build.py`
  - Web 编译页在执行 `model-opt build` 前也调用同样的检查

---

### 4) 关键根因 2：attention_mask 使用极端常数（约 -1e38）导致 TRT softmax 数值崩溃

你用于复现的 `attention_mask`（float32，4D）包含大量 `-2.3820e+38`：

- additive mask（加到 logits 上）
- 某些行/区域几乎全为该极端负值

在 TRT 的融合 kernel（matmul+add+softmax）与低精度路径下，这类极端值更容易触发：

- 中间 reduction/exp 溢出或出现 `inf/-inf`
- `inf - inf` 等未定义行为 → NaN
- NaN 扩散到后续 token 行与后续层

对照实验（来自 debug 脚本）：
- 当 mask 取 `torch.finfo(float32).min`/`-1e38` 量级时：
  - `last_hidden_state` 可能全 0 或出现 NaN
  - `present_key_values.1` 后半段 token 行出现 NaN
- 当 mask 改为温和的 `-1e4` 时：
  - `last_hidden_state nan=0 inf=0`
  - `present_key_values.0..N nan=0 inf=0`，深层稳定

结论：
**不要用 finfo.min/1e38 量级常数做 additive attention mask**（至少在 TRT 路径上）。
推荐：
- `mask_value = -1e4`（已验证稳定）

脚本侧支持参数：

```bash
PYTHONPATH=src python scripts/debug/dump_trt_engine_io.py \
  --engine /path/to/llm.engine \
  --seq_len 968 \
  --attention_mask_mode causal_float \
  --attention_mask_value=-1e4
```

#### 推理集成：`Pi05TensorRTExecutor` 对 `attention_mask` 的默认裁剪

OpenPI 训练/推理仍通过 `third_party/openpi/.../pi0_pytorch.py` 的 `_prepare_attention_masks_4d` 生成加性 mask（`0` / 约 `-2.3819763e38`）。该张量经 `gemma_pytorch.PaliGemmaWithExpertModel.forward` 原样传入 HF Gemma；**PyTorch 路径**下 attention 的 softmax 常在 **float32** 中计算，极端负 mask 一般仍可数值稳定。**TensorRT** 侧则更易在融合核（低精度）里溢出为 NaN。

因此在 **`src/model_optimizer/infer/tensorrt/pi05_executor.py`** 中，在调用 **`llm_engine`** / **`expert_engine`** 之前，对传入引擎的 `attention_mask` 执行：

- `attention_mask.clamp(min=neg_cap)`，默认 **`neg_cap = -1e4`**
- 与「禁止注意力」的语义一致（softmax 后仍为 0 概率），但避免 TRT 中 `-1e38` 量级导致的崩溃

可通过 executor 的 config（如 `addict.Dict`）设置：

- **`trt_attention_mask_neg_cap`**：浮点数，默认 **`-1e4`**；读取时使用 **`Mapping.get(...)`**（见 `_resolve_trt_attention_mask_neg_cap`）
- 设为 **`None`**：不做裁剪，便于与 PyTorch 或 debug 脚本做 A/B 对照

**注意（addict.Dict）**：对缺失键不要用 `getattr(config, "trt_attention_mask_neg_cap", -1e4)`——`addict` 会为缺失键**自动创建嵌套 `Dict`**，默认的 `-1e4` 永远不会被用到，且会得到 `TypeError: ... got 'Dict'`。未配置该项时应依赖 `.get` 分支得到默认 `-1e4`。

若你的入口未经过 `Pi05TensorRTExecutor` 而直接调用 `Engine`，应在**同样的边界**对 `attention_mask` 做等价处理。

---

## 与 TensorRT-Edge-LLM 的关键差异（为什么它更稳）

TensorRT-Edge-LLM 在 ONNX 导出上更“函数式”：

- KV cache 输入输出按层展开：
  - `past_key_values.{i}` / `present_key_values.{i}`
- 每层 `decoder_layer` 返回 `present_key_value`（显式张量）
- 避免依赖 `DynamicCache.update` 的 Python 副作用

此外，它在精度/插件契约上更严格（常见是 FP16 KV cache 与插件输入）。

---

## 结论与建议（工程落地）

### 必做
- **构建精度与 ONNX dtype 必须一致**
  - bf16 ONNX → bf16 build
  - fp16 ONNX → fp16 build
- **additive attention_mask 不要用 -1e38 量级**
  - 推荐 `-1e4`
- **走 `Pi05TensorRTExecutor` 时**：已对 LLM/Expert 引擎输入默认做 `clamp`（见上文「推理集成」）；自定义 TRT 入口需自行对齐

### 推荐
- 导出 KV cache 时使用 per-layer 显式输出（`present_key_values.{i}`）
  - 便于 debug、对齐 Edge-LLM、避免 DynamicCache 副作用问题
- 保留 `dump_trt_engine_io.py` 作为回归工具：
  - “engine 是否算对”可以一眼看出

---

## 相关文件索引

- 导出专用 forward / per-layer KV 输出：
  - `src/model_optimizer/models/pi05/llm.py`
- TRT engine 输入输出 dump：
  - `scripts/debug/dump_trt_engine_io.py`
- TRT build 精度一致性检查：
  - `src/model_optimizer/trt_build/build.py`
  - `src/model_optimizer/trt_build/cli.py`
  - `src/model_optimizer/webui/runners/build.py`
- PI0.5 TensorRT 推理（含 `attention_mask` 默认裁剪）：
  - `src/model_optimizer/infer/tensorrt/pi05_executor.py`
- TensorRT-Edge-LLM 参考实现：
  - `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/onnx_export/llm_export.py`
  - `third_party/TensorRT-Edge-LLM/tensorrt_edgellm/llm_models/models/llm_model.py`

