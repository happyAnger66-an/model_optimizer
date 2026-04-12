# Pi0.5 量化与部署对比分析（WebUI）

本文汇总 **LeRobot 评估 WebUI**（`scripts/deployment/pi05/lerobot_eval_webui_server.py`，实现包 `lerobot_eval_webui`）中与 **量化、TensorRT、ONNX Runtime、误差与分层分析** 相关的用法：多后端对比、PTQ 分层报告、以及基于 **NVIDIA Polygraphy** 的子图张量级对比。

更通用的 WebUI 启动、协议与标定采集见 [web_ui.md](web_ui.md)；**PTQ 双路 PyTorch 与 QuantLinear 分层报告**的字段与 JSON 格式见 [quantize_ptq_compare.md](quantize_ptq_compare.md)。

---

## 1. 环境与入口

### 1.1 `PYTHONPATH`

与 [web_ui.md](web_ui.md) 一致，至少需要 openpi 源码；使用 TensorRT / ONNX Runtime 子图引擎时还需 model_optimizer 的 `src`：

```bash
export PYTHONPATH="/path/to/model_optimizer/third_party/openpi/src:/path/to/model_optimizer/src:${PYTHONPATH}"
```

### 1.2 入口脚本

```bash
python scripts/deployment/pi05/lerobot_eval_webui_server.py \
  --checkpoint /path/to/checkpoint \
  --config pi05_libero \
  ...
```

参数定义见 `scripts/deployment/pi05/lerobot_eval_webui/config.py`（tyro：`Args`）。命令行一般为 **kebab-case**（如 `--trt-ort-compare`）。

### 1.3 可选依赖

| 场景 | 建议依赖 |
|------|-----------|
| Polygraphy 子图对比 | `polygraphy`、`onnx`、`onnxruntime-gpu`；与本机 TensorRT Python 包一致 |
| 安装方式 | `pip install polygraphy onnx onnxruntime-gpu`；或安装本仓库时带 `pi05` 可选依赖（`setup.py` 中 `pi05` 已含 `polygraphy>=0.49.22`） |

未安装 Polygraphy 时，开启 `--trt-ort-polygraphy-compare` 后，`meta.trt_ort_polygraphy` 会包含 `import_error` 提示，**不会**中断后续评估流。

---

## 2. 对比模式互斥关系

以下布尔标志 **至多开启一个**（由 `bundle.load_infer_bundle` 校验）：

| 标志 | 含义 |
|------|------|
| `--compare-mode` | PyTorch 浮点 vs TensorRT（第二路挂 TRT） |
| `--ptq-compare` | PyTorch 浮点 vs PyTorch 选择性 PTQ（fake quant） |
| `--ptq-trt-compare` | PyTorch PTQ vs TensorRT |
| `--ort-compare` | PyTorch 浮点 vs ONNX Runtime |
| `--trt-ort-compare` | TensorRT vs ONNX Runtime（主路 TRT，第二路 ORT） |

与 `inference_mode`、`engine-path`、`ort-engine-path` 及各 `*_engine` / `ort_*_engine` 的组合约束见下文各节。

---

## 3. PyTorch 浮点 vs TensorRT（`--compare-mode`）

- **要求**：`--inference-mode pytorch`（主路为 PyTorch）；`--engine-path` 与需要替换的子模块对应的 `--vit-engine` / `--llm-engine` / `--expert-engine` / `--denoise-engine` / `--embed-prefix-engine`（与单路 `tensorrt` 模式相同）。
- **行为**：加载两份 `create_trained_policy`；第二份挂载 TensorRT。WebSocket `step` 中带第二路预测与 `mse_trt`、`mse_pt_trt` 等（与 [web_ui.md](web_ui.md) 中 compare 语义一致）。
- **meta**：`compare_mode: true`，`pred1_name` / `pred2_name` 一般为 PT / TRT，`pair_name`：`PT−TRT`。

---

## 4. PyTorch 浮点 vs ONNX Runtime（`--ort-compare`）

- **要求**：`--inference-mode pytorch`；`--ort-engine-path` 与 `--ort-vit-engine` 等（与单路 `onnxrt` 一致）。
- **行为**：第二份 policy 挂载 ONNX Runtime 子图引擎。
- **meta**：`ort_compare: true`，`pair_name`：`PT−ORT`。

---

## 5. TensorRT vs ONNX Runtime（`--trt-ort-compare`）

### 5.1 用途

在 **同一 checkpoint、同一套观测** 下，一路用 **TensorRT 序列化引擎（`.engine`）**，另一路用 **ONNX Runtime（`.onnx`）**，流式对比动作预测与相对 GT 的误差；适用于部署前核对 TRT 与 ORT 是否一致。

### 5.2 要求

- `--engine-path`：TRT 引擎**目录**。
- `--vit-engine` / `--llm-engine` / …：TRT 侧各子图 **文件名**（如 `vit.engine`）。
- `--ort-engine-path`：ORT 所用 ONNX **目录**。
- `--ort-vit-engine` / `--ort-llm-engine` / …：ORT 侧各子图 **文件名**（如 `vit.onnx`）。
- 子图成对配置：某一子模块只有在 **TRT 名与 ORT 名均非空** 时才会在双路推理中替换该子模块（与 `bundle` / `tensorrt_backend` / `onnxrt_backend` 行为一致）。

### 5.3 meta 与前端标签

- `trt_ort_compare: true`
- `pred1_name`：`TRT`，`pred2_name`：`ORT`，`pair_name`：`TRT−ORT`
- `meta.tensorrt` / `meta.onnxrt` 中会分别记录两套路径与引擎文件名，便于复现。

---

## 6. Polygraphy 子图张量对比（与 `--trt-ort-compare` 联用）

在 **TRT vs ORT** 评估之外，可在 **加载阶段** 对子图 ONNX 再跑一轮 **Polygraphy** 风格的对比，将结果写入 **`meta.trt_ort_polygraphy`**，便于定位子图内部哪路输出差异大。

实现见 `scripts/deployment/pi05/lerobot_eval_webui/trt_ort_polygraphy_compare.py`；思路对齐 Model-Optimizer 中 `ReferenceRunner`：`ModifyOutputs(..., MARK_ALL)` + `OnnxrtRunner` + `polygraphy.comparator.Comparator`（见上游 `referencerunner` 一类用法）。

### 6.1 开启方式

在已配置 `--trt-ort-compare` 及上述 TRT/ORT 路径的前提下，增加：

```bash
--trt-ort-polygraphy-compare
```

### 6.2 命令行参数一览

| 参数 | 默认 | 含义 |
|------|------|------|
| `--trt-ort-polygraphy-compare` | `False` | 加载完成后、推送 meta 前执行 Polygraphy 对比 |
| `--trt-ort-polygraphy-mark-all` | `False` | 使用 `MARK_ALL` 将 ONNX 中间张量提升为图输出 |
| `--trt-ort-polygraphy-rebuild-trt` | `False` | 与 `mark_all` 联用：从 MARK_ALL 后的 ONNX **现场编译** TensorRT，再与 ORT 对比 |
| `--trt-ort-polygraphy-parts` | 空 | 仅跑指定子图：`vit`、`embed_prefix`、`llm`、`expert`、`denoise`；不传则处理 **所有已同时配置 TRT+ORT 文件名** 的子图 |
| `--trt-ort-polygraphy-ort-providers` | `CUDAExecutionProvider` `CPUExecutionProvider` | ORT 执行提供程序顺序 |
| `--trt-ort-polygraphy-max-report-tensors` | `256` | 每个子图写入 meta 的张量摘要条数上限（按 `max_abs` 降序截断） |
| `--trt-ort-polygraphy-seed` | `0` | 合成输入的随机种子基值（各子图会再做稳定偏移） |

**约束**：若 `--trt-ort-polygraphy-mark-all` 为真，则 **必须** `--trt-ort-polygraphy-rebuild-trt`，否则会在报告中返回明确错误（预序列化 `.engine` 与 MARK_ALL 后的图输出集合不一致，无法直接对齐）。

### 6.3 两种运行语义

| 模式 | 条件 | 说明 |
|------|------|------|
| **默认** | `mark_all=False` | 在 **原始 ONNX 的声明输出** 上，对比 **CUDA EP 的 ORT** 与 **磁盘上的 `.engine`**。要求引擎由 **同源 ONNX** 构建且 **I/O 绑定名** 与 ONNX 输出名一致。 |
| **全量张量（类逐层）** | `mark_all=True` 且 `rebuild_trt=True` | 对 MARK_ALL 后的 ONNX：ORT 与 **TensorRT Builder 当场编译** 的引擎对比，可得到大量中间张量统计；**不使用**你部署目录里的预编译 `.engine`。大子图（如整图 LLM）可能 **耗时很长或 OOM**，建议配合 `--trt-ort-polygraphy-parts vit` 等逐步排查。 |

### 6.4 输入说明

当前实现使用 **基于各子图输入元数据的合成随机输入**（`float32` / `int32` / `int64` 等按元数据选择），用于 **连通性与数量级** 级别的对齐检查；**不代表** LeRobot 真实帧上的数值误差。若需与真实数据一致，需后续扩展（例如从 Polygraphy JSON / NPZ 加载 feeds）。

### 6.5 `meta.trt_ort_polygraphy` 结构（概要）

- `enabled` / `ok`：是否执行及是否全部子图成功。
- `mark_all` / `rebuild_trt` / `ort_providers`：本次配置快照。
- `subgraphs`：按子图标签（如 `vit`、`llm`）分组的字典。每个子图通常含：
  - `onnx`、`engine`（默认模式下为 TRT 引擎绝对路径；`mark_all+rebuild` 时 `engine` 可为 `null`）。
  - `ok`：本子图是否成功。
  - `tensors`：列表项含 `name`、`shape`、`max_abs`、`mean_abs`、`rmse` 等；若形状不一致则含 `shape_left` / `shape_right` / `error`。
  - 失败时含 `error` 或顶层 `import_error` / `hint`。

---

## 7. PyTorch PTQ 与分层分析（`--ptq-compare` / `--ptq-trt-compare`）

### 7.1 `--ptq-compare`

- **要求**：`--inference-mode pytorch`；`--ptq-quant-cfg`、`--ptq-calib-dir`、`--ptq-parts`（非空，且为 `vit` / `llm` / `expert` / `denoise` 的子集）。
- **可选**：`--ptq-layer-report-path` 等生成分层 JSON，并嵌入 `meta.ptq_layer_report`（若读取成功）。
- **详细参数、管线、Layer JSON 字段**：见 [quantize_ptq_compare.md](quantize_ptq_compare.md)。

### 7.2 `--ptq-trt-compare`

- **一路**：同一份 policy 上就地选择性 PTQ（fake quant）。
- **二路**：另一份 policy 挂载 TensorRT。
- **meta**：`pred1_name=PTQ`、`pred2_name=TRT`、`pair_name=PTQ−TRT`；仍需 `--ptq-quant-cfg`、`--ptq-calib-dir`、`--ptq-parts` 与 `--engine-path` 及各 `*_engine`。

---

## 8. WebSocket `meta` 中与分析相关的字段索引

| 字段 | 出现条件 | 用途 |
|------|-----------|------|
| `ptq_compare` / `ptq_parts` / `ptq_quant_cfg` / `ptq_calib_dir` | PTQ 相关 | PTQ 对比与校准路径说明 |
| `ptq_layer_report_path` / `ptq_layer_report` | 配置了 layer 报告 | 分层误差与可选直方图 |
| `compare_mode` / `ort_compare` / `trt_ort_compare` | 对应模式 | 前端与下游区分对比类型 |
| `pred1_name` / `pred2_name` / `pair_name` | 各双路模式 | UI 标签 |
| `tensorrt` / `onnxrt` | TRT 或 ORT 参与时 | 引擎目录与文件名快照 |
| `trt_ort_polygraphy` | `--trt-ort-polygraphy-compare` | Polygraphy 子图对比摘要 |

完整协议与 `step` 指标见 [web_ui.md](web_ui.md)。

---

## 9. 示例命令

### 9.1 TRT vs ORT 评估 + Polygraphy（仅声明输出，对比预置 engine）

```bash
export PYTHONPATH="third_party/openpi/src:src:${PYTHONPATH}"

python scripts/deployment/pi05/lerobot_eval_webui_server.py \
  --checkpoint /path/to/checkpoint \
  --config pi05_libero \
  --dataset-root ~/.cache/huggingface/lerobot/... \
  --trt-ort-compare \
  --engine-path /path/to/trt_engines \
  --vit-engine vit.engine \
  --llm-engine llm.engine \
  --expert-engine expert.engine \
  --denoise-engine denoise.engine \
  --ort-engine-path /path/to/onnx_models \
  --ort-vit-engine vit.onnx \
  --ort-llm-engine llm.onnx \
  --ort-expert-engine expert.onnx \
  --ort-denoise-engine denoise.onnx \
  --trt-ort-polygraphy-compare \
  --trt-ort-polygraphy-parts vit \
  --host 0.0.0.0 --port 8765
```

### 9.2 Polygraphy MARK_ALL + 现场编译 TRT（慎用，子图宜小）

```bash
... \
  --trt-ort-compare \
  ... \
  --trt-ort-polygraphy-compare \
  --trt-ort-polygraphy-mark-all \
  --trt-ort-polygraphy-rebuild-trt \
  --trt-ort-polygraphy-parts vit
```

---

## 10. 相关代码路径

| 路径 | 说明 |
|------|------|
| `scripts/deployment/pi05/lerobot_eval_webui_server.py` | CLI 薄入口 |
| `scripts/deployment/pi05/lerobot_eval_webui/config.py` | 全部 tyro 参数 |
| `scripts/deployment/pi05/lerobot_eval_webui/bundle.py` | 加载策略、校验互斥、组装 `meta` |
| `scripts/deployment/pi05/lerobot_eval_webui/chunk_infer.py` | 单 chunk 推理与 `step` 指标 |
| `scripts/deployment/pi05/lerobot_eval_webui/infer_backends.py` | 双路后端选择（含 `TrtOrtCompareBackend`） |
| `scripts/deployment/pi05/lerobot_eval_webui/ptq_compare.py` | PTQ 与分层报告 |
| `scripts/deployment/pi05/lerobot_eval_webui/trt_ort_polygraphy_compare.py` | Polygraphy 子图对比 |
| `scripts/deployment/pi05/lerobot_eval_webui/tensorrt_backend.py` / `onnxrt_backend.py` | 引擎挂载 |

量化与标定在训练/离线流水线中的用法见 [pi05_quantize.md](pi05_quantize.md) 与 [eval.md](eval.md)。

---

## 11. 注意事项

1. **显存与耗时**：双 policy、Polygraphy `MARK_ALL` + `rebuild_trt`、或 PTQ layer report 都会显著增加资源占用与启动时间。
2. **引擎与 ONNX 一致性**：TRT vs ORT 流式评估与 Polygraphy「默认模式」均假设子图 **导出同源**；版本或 fusion 不一致时，数值差异可能来自图本身而非实现 bug。
3. **Polygraphy 与 WebUI 推理独立**：`trt_ort_polygraphy` 在 **加载阶段** 完成；其合成输入 **不** 替代 LeRobot 上的 `policy.infer` 数据流。

与 [web_ui.md](web_ui.md) 配合阅读：前者侧重通用启动与协议，本文侧重量化与多后端对比分析。
