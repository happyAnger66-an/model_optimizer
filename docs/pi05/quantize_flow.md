# pi05 量化标定流程与 Fake Quantization（QDQ）说明

> 本文说明 `quantize_model` → `mtq.quantize` 中标定阶段 **为何使用 fake quantization**，并以 pi05 真实 **Linear 层** 为例走通全过程。  
> 流水线各 stage 的 export / quantize / build 分工见 [pipeline.md](./pipeline.md)；入口代码见 `src/model_optimizer/quantization/quantization_utils.py`。

---

## 目录

1. [Fake quant 是什么](#一fake-quant-是什么)
2. [为什么要 fake quant（而非纯 FP32 标定）](#二为什么要-fake-quant而非纯-fp32-标定)
3. [与真量化的区别](#三与真量化的区别)
4. [代码入口：`quantize_model` 与第 331 行](#四代码入口quantize_model-与第-331-行)
5. [实例：Gemma `q_proj` 一层 Linear 的完整过程](#五实例gemma-q_proj-一层-linear-的完整过程)
6. [AWQ 等算法与 fake quant](#六awq-等算法与-fake-quant)
7. [与 pi05 各 stage 的关系](#七与-pi05-各-stage-的关系)
8. [关键参考](#八关键参考)

---

## 一、Fake quant 是什么

在 `mtq.quantize`（NVIDIA ModelOpt）里，会在 `nn.Linear` / `nn.Conv2d` 等旁插入 **`TensorQuantizer`**。前向路径大致为：

```
x (fp16/bf16/fp32)
  → 按当前 scale 量化到 INT8/FP8 等
  → 立刻反量化回浮点
  → 再送进 Linear / MatMul
```

数学上仍是 **浮点张量**在流，但数值上已经带了一次「量化再还原」的误差。这就是 **fake quantization（QDQ：Quantize → Dequantize）**，与 TRT/ONNX 里的 **QuantizeLinear + DequantizeLinear** 是同一类思想。

改造后的模块概念结构：

```
原来:  x → Linear(W) → y

改造后:
  x → [input_quantizer]  → x̂ (QDQ 后)
      → Linear(Ŵ)        → y
      （权重侧有 weight_quantizer，对 W 做 QDQ）
```

标定阶段 **不跑整数 GEMM**；部署时 TRT 等后端才用 FP8/INT8 **真算子**。

---

## 二、为什么要 fake quant（而非纯 FP32 标定）

### 2.1 标定的是「会被量化的那个张量」

Max / percentile 等标定器挂在 `TensorQuantizer` 上，统计的是 **经过 QDQ 之后、或即将被量化那一截** 的动态范围（amax），而不是「假如永远不量化」的 FP32 极值。

若只用 FP32 前向取激活 max 再硬套 scale，与真实 QDQ 路径不完全一致。

### 2.2 后面层看到的是「前面量化后的输出」（误差传播）

部署时数据流是：

```
Linear1 → QDQ → Linear2 → QDQ → Linear3 → ...
```

若标定 **全程 FP32**，`k_proj` 收集的激活来自 **无误差** 的 `q_proj` 输出；部署时 `k_proj` 的输入却是 **带量化噪声** 的。两者分布会有偏差。

Fake quant 标定时，**上一层 QDQ 后的输出** 作为下一层输入，后面 Linear 收集到的统计更接近 **链式量化后的真实分布**。  
因此：**既是为本层定准 scale，也是让后续层在标定阶段就看到「使用量化后的分布」**——两层目的都有。

### 2.3 部分算法必须在 QDQ 图里跑

| 方法 | 为何需要 fake quant |
|------|---------------------|
| **Max / percentile** | 在 quantizer 上直接统计待量化张量 |
| **AWQ（awq_lite 等）** | 在 QDQ 环境下搜索 `pre_quant_scale`；LLM 标定常临时转 fp32（`llm.py`） |
| **SmoothQuant 等** | 依赖带 quantizer 的前向 |

### 2.4 与导出一致

标定结束后 scale 固定；`export` 时 ONNX/TRT 图里就是 **QDQ 节点 + 同一套 scale**。标定若不用 QDQ，导出图与标定假设容易不一致。

### 2.5 代码里的佐证

`measure_quant_error=True` 时，`_tensor_qdq_error_analysis` 对每个 `TensorQuantizer` 比较 **输入 vs 输出**（量化后再反量化相对原输入差多少），量化的是 **单点 fake quant 误差**；整条链路则靠标定循环里 **每层都走 QDQ** 来对齐。

---

## 三、与真量化的区别

| | Fake quant（标定 / PyTorch） | 真量化（TRT / Engine） |
|--|------------------------------|-------------------------|
| 中间算子 | 仍是 FP32/BF16 **MatMul** | INT8/FP8 **Gemm** 等 |
| 目的 | 定 scale、搜权重、评估 QDQ 误差 | 低延迟、省带宽 |
| 数值 | 浮点算子 + 模拟量化噪声 | 低精度硬件路径 |

Fake quant **不是为了在 PyTorch 里永久用整数算子**，而是在仍用浮点 kernel 的前提下 **复现量化误差，并把 scale 定准**。

---

## 四、代码入口：`quantize_model` 与第 331 行

### 4.1 调用链（以 Vit 为例）

```
Vit.quantize(quant_cfg, calib_data, export_dir)
  ├─ get_calibrate_dataset(calib_data)     # pi05_vit / pi05_llm 等标定数据
  ├─ quantize_model(self, quant_cfg, loader)
  │    ├─ 构造 calibrate_loop / _run_calib_forward
  │    ├─ mtq.quantize(model, quant_config, forward_loop=calibrate_loop)  ← 核心
  │    ├─ mtq.print_quant_summary(model)
  │    └─ 可选 _tensor_qdq_error_analysis
  ├─ set_dynamic_quant(self, "bf16")
  └─ export(export_dir) → vit.onnx / llm.onnx / ...
```

配置来源：`src/model_optimizer/quantization/cfg.py`（如 `"fp8": mtq.FP8_DEFAULT_CFG`）。

### 4.2 `calibrate_loop` 做什么

```python
def calibrate_loop(qmodel):
    for data in calib_dataloader:
        _run_calib_forward(qmodel, data)   # model(**batch) 或 forward_context(...)
```

ModelOpt 在标定阶段 **反复调用** 该回调，用真实 calib 数据跑前向，收集激活 amax / AWQ 统计等。

### 4.3 `mtq.quantize` 内部典型阶段

```mermaid
flowchart TB
  A["mtq.quantize(model, quant_config, forward_loop)"] --> B["解析 quant_config"]
  B --> C["插入 TensorQuantizer / QuantLinear"]
  C --> D{"根模块是 HF PreTrainedModel?"}
  D -->|是| E["register HF attention bmm_quantizer 等"]
  D -->|否| F["跳过"]
  E --> G["多次 forward_loop(qmodel)"]
  F --> G
  G --> H["收集 amax / AWQ 等"]
  H --> I["固化 scale，fake-quant 就绪"]
```

标定在 **已插入 quantizer 的同一棵模块树** 上原地（in-place）完成；**不导出 ONNX**（导出在调用方 `quantize()` 里的 `export()`）。

### 4.4 标定前向与 dtype

| 路径 | 行为 |
|------|------|
| 默认 | `_calib_batch_to_model_device_dtype`：张量移到 GPU，浮点对齐到 `next(model.parameters()).dtype` |
| `forward_context`（denoise + FP8_KV） | 只 `.to(device)`，避免 `x_t` 与 fp32 `action_in_proj` dtype 冲突 |

标定数据：`open_pi05_calib_for_quantize` → `list[dict]` 或分片流式；key 需与对应 `forward` 一致（如 `pixel_values`、`inputs_embeds`）。

---

## 五、实例：Gemma `q_proj` 一层 Linear 的完整过程

以 **Gemma 2B 第一层 `layers.0.self_attn.q_proj`** 为例（`nn.Linear(2048, 2048, bias=False)`），配置 **`mtq.FP8_DEFAULT_CFG`**（与 GPU `llm_kv_fp8` 同族）。

### 5.1 标定前：纯浮点

```
x:  [B, L, 2048]   # prefix embedding，L≈768~968
W:  [2048, 2048]
y = x @ W^T        # [B, L, 2048]
```

### 5.2 图改造：插入 quantizer

ModelOpt 将 `q_proj` 变为带 **`input_quantizer`**、**`weight_quantizer`** 的 `QuantLinear`（名称以 `print_quant_summary` 为准）。

| 对象 | FP8 E4M3 典型方式 | scale 来源 |
|------|-------------------|------------|
| 权重 W | per-tensor FP8 | W 的 absmax → `s_w` |
| 激活 x | per-tensor FP8 | calib 循环 fake QDQ 路径上的 amax → `s_x` |

### 5.3 一个 calib batch 内的 fake quant 前向

**（1）激活 input_quantizer**

\[
s_x = \frac{\max(|x|)}{F_{\max}}, \quad F_{\max} \approx 448 \text{（E4M3）}
\]

跨多个 calib 样本对 amax 取 max（`algorithm: max`），得到该层固定的 `s_x`。

\[
\hat{x} = \mathrm{DQ}(\mathrm{Q}(x / s_x)) \cdot s_x
\]

**（2）权重 weight_quantizer**

\[
s_w = \frac{\max(|W|)}{F_{\max}}, \quad
\hat{W} = \mathrm{DQ}(\mathrm{Q}(W / s_w)) \cdot s_w
\]

**（3）仍用浮点 MatMul**

```
y = matmul(x̂, Ŵ^T)    # bf16 GEMM，非整数 kernel
```

`k_proj`、`v_proj` 等 **下一层** 的 `input_quantizer` 观测到的是带误差的 `y`，与部署链一致。

### 5.4 缩小维度的数值直觉

设 `x: [1,4]`，`W: [4,4]`，\(F_{\max}=448\)：

```
max|W| = 2.0  →  s_w = 2.0/448
max|x| = 2.1  →  s_x = 2.1/448
x̂ = QDQ(x),  Ŵ = QDQ(W)
y = x̂ @ Ŵ^T   # 与 y_fp = x @ W^T 有舍入误差
```

真实 `q_proj` 为 `[B,L,2048]`，per-tensor 表示 **整层共用一个 `s_x`、一个 `s_w`**；calib 在 **所有样本、所有 token** 上合并 amax（对齐文档 O7 act_scales 思路）。

### 5.5 标定结束 → 导出

`mtq.print_quant_summary` 打印各层 quantizer 与 scale。  
`LLM.export()` → ONNX 中该 Linear 变为：

```
QuantizeLinear(x, scale=s_x) → DequantizeLinear
QuantizeLinear(W, scale=s_w) → DequantizeLinear
→ MatMul/Gemm  （TRT 可融合为 FP8 GEMM）
```

**标定 fake quant 与导出 QDQ 使用同一套 scale**。

### 5.6 TensorRT 部署（真量化）

Engine 内用 **FP8/BF16 混合 GEMM** 与 `s_x`、`s_w` 在硬件上执行；PyTorch 标定阶段只负责 **定 scale**，不跑 TRT 快路径。

### 5.7 单层时序总览

```mermaid
sequenceDiagram
  participant Calib as calib_dataloader
  participant MO as mtq.quantize
  participant Q as q_proj QuantLinear
  participant Exp as export ONNX

  MO->>Q: 插入 input/weight TensorQuantizer
  loop 每个 calib batch
    Calib->>Q: forward 经过 q_proj
    Q->>Q: 更新 input amax（fake QDQ 路径）
  end
  MO->>Q: 固化 s_x, s_w
  Exp->>Q: QDQ 节点写入 llm.onnx
  Note over Exp: TRT build → 真 FP8 GEMM
```

---

## 六、AWQ 等算法与 fake quant

`quant_cfg` 为 **`int4_awq`** 时（`cfg.py` → `mtq.INT4_AWQ_CFG`）：

1. 仍插入 `TensorQuantizer`（W4 等）
2. `forward_loop` **多轮**前向，在 **QDQ 环境**下搜索 `pre_quant_scale`
3. LLM 路径可能先将模型临时转 **fp32** 标定（`llm.py`），再恢复原 dtype 导出

AWQ **不能**在纯 FP32 图上搜索，否则搜到的权重对部署无效。

---

## 七、与 pi05 各 stage 的关系

| Stage | `quantize_model` 的 `model` 根 | 标定前向 | 量化配置示例 |
|-------|-------------------------------|----------|--------------|
| **vit** | 整个 `Vit` | `self(pixel_values)` | `fp8` / `nvfp4` |
| **llm** | HF `GemmaModel`（`self.model`） | `model(**{inputs_embeds, attention_mask, ...})` | `fp8` + 可选 `FP8_KV_CFG` |
| **denoise** | `Pi05DenoiseStep` 或 `gemma_expert` + `forward_context` | 完整 denoise `forward` | `nvfp4` / FP8_KV 等 |
| **expert** | 同 expert 子模块 | 依 collector | 同族 |

标定完成后各 stage **`export()` 产出 ONNX**，再 TRT build；TVM 路径见 [how_to_use_tvm.md](./how_to_use_tvm.md)（MVP 可先 FP16，INT8 复用 calib 思路）。

---

## 八、关键参考

| 资源 | 路径 |
|------|------|
| 量化入口 | `src/model_optimizer/quantization/quantization_utils.py` — `quantize_model`，331 行 `mtq.quantize` |
| 量化配置 | `src/model_optimizer/quantization/cfg.py` — `QUANT_CFG_CHOICES` |
| LLM 量化 | `src/model_optimizer/models/pi05/llm.py` — `quantize()` |
| Vit 量化 | `src/model_optimizer/models/pi05/vit.py` — `quantize()` |
| Denoise 量化 | `src/model_optimizer/models/pi05/dit.py` — `quantize()` |
| 标定数据 | `src/model_optimizer/calibrate/pi05_calib_load.py` |
| Pipeline 分工 | [pipeline.md](./pipeline.md) |

---

## 附录：一句话结论

**Fake quantization** 在标定阶段用 **QDQ + 浮点 MatMul** 模拟部署时的量化误差，为每层定准 **scale**（及 AWQ 等所需的权重变换），并使 **后续层统计到的激活分布** 与 TRT/ONNX **真量化推理** 一致；标定结束后 **同一套 scale** 写入导出图，由 TensorRT 等在运行时执行 **真 FP8/INT8 GEMM**。

---

*文档整理自 pi05 量化标定讨论。*
