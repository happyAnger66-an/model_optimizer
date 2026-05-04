# INT4 AWQ 量化流程说明

本文说明在 **model_optimizer** 中以 **Pi05 LLM（Gemma 解码器）** 为例，使用 NVIDIA **ModelOpt**（`modelopt.torch.quantization`）做 **INT4 AWQ（W4A16，权重 4bit、激活保持高精度）** 时的端到端流程。算法实现以 ModelOpt 源码为准；本仓库在 LLM 包装层增加了与 dtype 相关的适配。

---

## 1. 在做什么（结果形态）

| 项目 | 说明 |
|------|------|
| 权重量化 | 4bit **块静态**量化，默认沿权重最后一维块长 **128**（`block_sizes: {-1: 128, "type": "static"}`） |
| 激活 | `*input_quantizer` 默认 **关闭** → 推理侧为 **W4A16**（权重 int4 fake-quant，激活不降 bit） |
| 校准算法 | `awq_lite`：用校准数据估计激活/权重尺度，在多个 **α** 上搜索平滑尺度，再写入权重侧的平滑（与 `pre_quant_scale` 相关逻辑） |

与纯 `algorithm: "max"` 的块量化相比，AWQ 多了一段 **基于校准数据的尺度搜索**，通常需要 **完整跑两遍**（见下文）校准集上的前向。

---

## 2. 配置从哪里来

仓库示例：`config/quant/llm_quant_int4_awq.py`

```python
from modelopt.torch.quantization import INT4_AWQ_CFG
QUANT_CFG = copy.deepcopy(INT4_AWQ_CFG)
```

ModelOpt 内置的 `INT4_AWQ_CFG` 核心字段等价于（语义摘要）：

- `quant_cfg["*weight_quantizer"]`：`num_bits=4`，`block_sizes` 最后一维 128、静态块，`enable=True`
- `quant_cfg["*input_quantizer"]`：`enable=False`（W4A16）
- `algorithm`：`{"method": "awq_lite", "alpha_step": 0.1}`（α 从 0 到 1 步进 0.1）

其他量化子模块（如 KV 等）由 `INT4_AWQ_CFG` 自带的默认关闭项覆盖，具体以 ModelOpt 包内定义为准。

---

## 3. CLI 到本仓库代码的路径

1. **`model-opt quantize`**（`src/model_optimizer/quantization/cli.py`）  
   - 解析 `--quantize_cfg`，`get_quant_cfg()` 动态加载 Python 配置文件，得到 `QUANT_CFG` 字典。  
   - `get_model_cls(model_name)` 构造 **`LLM`**（`src/model_optimizer/models/pi05/llm.py`）。

2. **`LLM.quantize(quant_cfg, calib_data, export_dir, ...)`**  
   - `get_calibrate_dataset(calib_data)` → `open_pi05_calib_for_quantize(..., component="pi05_llm")`，得到可迭代校准样本（dict：`inputs_embeds`、`attention_mask`、`position_ids` 等）。  
   - 若 `quant_cfg` 的 `algorithm` 属于 **AWQ 族**（`method` 以 `awq` 开头，或字符串算法名以 `awq` 开头），则将 **`self.model`（HF Gemma 解码器）临时转为 `float32`**，再调用 `quantize_model`；结束后 **`finally` 中恢复**为原来的 `bfloat16` / `float16`。  
   - 目的：缓解 **bf16 + `awq_lite`** 在部分 GPU/版本上出现的数值或 CUDA 内核问题（例如 device-side assert，栈常落在 `TensorQuantizer`）。  
   - `quantize_model` 返回后：`set_dynamic_quant(self, "bf16")`（主要影响 NVFP4 路径；INT4 AWQ 一般无 NVFP4 线性）。  
   - 随后 **`export`** 写出 `llm.onnx` 等。

3. **`quantize_model`**（`src/model_optimizer/quantization/quantization_utils.py`）  
   - 定义 **`calibrate_loop(model)`**：对校准集每条样本 `data`，将张量移到 `model.device`，调用 **`model(**data, **kwargs)`**。  
   - 调用 **`mtq.quantize(model, quant_config, forward_loop=calibrate_loop)`**。

4. **`LLM.forward`**（同一 `llm.py`）  
   - 若解码器因 AWQ 已变为 **`float32`**，而校准张量仍为 **`bfloat16`**，则 **`F.linear` 会报 dtype 不一致**。  
   - 因此在进入 `self.model(...)` 前，用 **`self.model.dtype`**（若无则 `next(self.model.parameters()).dtype`）将 **`inputs_embeds` 转成与解码器一致的 dtype**。  
   - `attention_mask` / `position_ids` 保持原样（与 HF Gemma 约定一致）。

---

## 4. ModelOpt 内部：`mtq.quantize` 在 INT4 AWQ 下做什么

整体顺序可概括为：

1. **`apply_mode(..., mode=[("quantize", config)])`**  
   按 `quant_cfg` 把目标 `nn.Linear` 等替换为 **Quant** 模块（含 `weight_quantizer` / `input_quantizer`），并挂上校准所需状态。

2. **`calibrate(model, config["algorithm"], forward_loop=...)`**  
   对 `algorithm={"method": "awq_lite", ...}` 会进入 **`awq_lite`**（`modelopt/torch/quantization/model_calib.py`）。

### 4.1 `awq_lite` 对每个被量化的 Linear（概念流程）

对每个启用 `weight_quantizer` 的量化线性层，会安装 **`AWQLiteHelper`** 并 **替换 `forward`**，大致经历：

**阶段 A — Cache（`cache_mode = True`）**

- 打日志：`awq_lite: Caching activation statistics...`
- 调用 **`forward_loop(model)`**（即本仓库的 **`calibrate_loop`**：遍历校准集，**整模型前向**）。
- 在 patched `forward` 里：先关掉 `weight_quantizer`，清掉 `pre_quant_scale`，用 **`_forward_no_awq`** 得到 **未做 AWQ 权重量化** 的输出 `out_actual`；再打开 `weight_quantizer`。  
  在 cache 模式下：用 **`get_act_scale`** 累积 **`act_scale`**（与输入经 `input_quantizer` 后的统计有关；Pi05 INT4 AWQ 配置下输入量化通常关闭，行为以 ModelOpt 为准），并增加 `num_cache_steps`。
- 全模型跑完一遍后：**`max_calibrate(model, ...)`** 消化本阶段收集的统计（如部分 quantizer 的 amax）。

**阶段 B — 归一化激活尺度**

- 对每个参与 AWQ 的模块：若 `num_cache_steps > 0`，将 **`act_scale` 除以步数** 做平均；若出现 **NaN** 则 **对该层禁用 AWQ**，后续退化为对该层权重做 **`max_calibrate`**。  
- 可选：在分布式下对 **`act_scale`** 做 **DP 同步**。

**阶段 C — Search（`cache_mode = False`）**

- 打日志：`awq_lite: Searching parameters...`
- **再次**调用 **`forward_loop(model)`**（第二遍完整校准前向）。  
- 在 patched `forward` 里：对每个 **α ∈ {0, 0.1, …, 1.0}**：  
  - 用预计算的 **`act_scale`、`weight_scale`**（由权重分块 **`get_weight_scale`** 得到）通过 **`get_scale`** 得到 **`awq_scale`**；  
  - 设置 **`input_quantizer.pre_quant_scale`** 与 **`weight_quantizer.pre_quant_scale`**（与平滑/尺度搜索相关）；  
  - 再 **`_forward_no_awq`** 得到带平滑尝试的输出，与 **`out_actual`** 比 **MSE**，累加到 `loss[alpha]`。  
- 因此：**校准集至少会被完整前向两遍**（cache 一遍 + search 一遍），且 search 阶段每层在每个 batch 上可能多次前向（随 α 个数）。

**阶段 D — Postprocess**

- **`update_best_params`**：取 **`loss` 最小的 α**，得到 **`best_scale`**。  
- 去掉临时的 **`_pre_quant_scale`**；若配置了输入侧平滑相关逻辑则做相应收尾。  
- 若该层 AWQ 仍启用：**`apply_pre_quant_scale_and_smooth`**，把最优尺度作用到权重/平滑流程；否则对该层 **`max_calibrate`** 仅按权重做 max 校准。  
- **`cleanup`**：恢复原始 `forward`，删除 `awq_lite` 辅助对象等。

以上细节以你环境中的 **`modelopt`** 版本为准；不同版本行号可能略有差异。

---

## 5. 校准数据（Pi05 LLM）

- **来源**：`Pi05LLMCalibCollector` 钩子挂在 `language_model.forward` 上，保存与 **`LLM.forward`** 一致的 keyword：`inputs_embeds`、`attention_mask`、`position_ids`（见 `src/model_optimizer/calibrate/collector/pi05.py`）。  
- **加载**：`open_pi05_calib_for_quantize` 支持单文件 `.pt`、manifest + 分片、或 `pi05_llm_calib_shards` 目录（见 `src/model_optimizer/calibrate/pi05_calib_load.py`）。  
- **建议**：张量应为有限值；**`position_ids`** 与 **`attention_mask`** 需与训练/推理时一致，且与 **`max_position_embeddings`**、序列长度相容，否则可能在 Gemma 内部先触发错误，异步表现为后续 CUDA 断言。

---

## 6. 约束与排错要点

| 问题 | 建议 |
|------|------|
| Linear **`in_features % 128 != 0`** | 修改 `quant_cfg` 中 `*weight_quantizer` 的 `block_sizes[-1]` 为能整除的块长（如 64），否则 INT4 块内核或 reshape 可能失败。 |
| **`BFloat16 != float`** | 确保使用含 **`LLM.forward` 中 `inputs_embeds` 与 `model.dtype` 对齐** 的版本；AWQ 标定期间解码器为 fp32 时，输入不能仍为 bf16。 |
| CUDA **device-side assert** 且栈在 `TensorQuantizer` | 先 **`CUDA_LAUNCH_BLOCKING=1`** 看真实首错；检查校准数据 NaN/Inf 与 mask/position；确认已启用 **AWQ 标定期 fp32 解码器** 逻辑。 |
| 不想用 AWQ | 改用 **`INT4_BLOCKWISE_WEIGHT_ONLY_CFG` + `"algorithm": "max"`**，仅需（或主要）权重统计，行为更简单。 |

---

## 7. 相关文件索引

| 路径 | 作用 |
|------|------|
| `config/quant/llm_quant_int4_awq.py` | INT4 AWQ 量化配置入口 |
| `src/model_optimizer/models/pi05/llm.py` | `LLM.quantize`、`forward`、AWQ dtype 适配 |
| `src/model_optimizer/quantization/quantization_utils.py` | `quantize_model` / `calibrate_loop` |
| `src/model_optimizer/calibrate/collector/pi05.py` | `pi05_llm` 校准采集 |
| `src/model_optimizer/calibrate/pi05_calib_load.py` | 校准数据加载 |
| ModelOpt 包内 | `INT4_AWQ_CFG`、`awq_lite`、`mtq.quantize` |

更通用的 Pi05 量化命令与目录约定见 **`docs/pi05_quantize.md`**。
