# model_optimizer 项目架构与实现分析

## 1. 入口与 CLI 流程

### 1.1 入口注册

```
model_optimizer-cli  /  model-opt  →  setup.py entry_points  →  model_optimizer.cli:main
```

- `setup.py` 中注册了 `model-optimizer-cli` 和 `model-opt`，均指向 `model_optimizer.cli:main`
- `cli.py` 的 `main()` 调用 `launcher.launch()`
- `launcher.py` 的 `launch()` 解析子命令并 lazy 导入对应 CLI

### 1.2 命令分发

```
launcher.launch()
  │
  ├─ webui         → webui.interface.run_web_ui
  ├─ quantize      → quantization.cli.quantize_cli
  ├─ export        → convert.convert_formt.convert_model
  ├─ calibrate     → calibrate.cli.calibrate_cli
  ├─ build         → trt_build.cli.build_cli
  ├─ eval          → evaluate.cli.eval_cli
  ├─ profile       → profile.cli.profile_cli
  ├─ download      → download.cli.download_cli
  ├─ compare       → compare.cli.compare_cli
  ├─ datasets      → datasets.cli.eval_datasets
  └─ version       → 打印版本信息
```

---

## 2. 整体架构

### 2.1 分层结构

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / WebUI 层                           │
│  (quantize, export, calibrate, build, eval, profile, compare)   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      业务逻辑 / 服务层                            │
│  quantization, convert, trt_build, evaluate, calibrate, ...     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      模型抽象层 (registry + Model)                 │
│  get_model_cls(model_name) → Model.construct_from_name_path()   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      具体模型实现 (Vit, LLM, Expert, Yolo)         │
│  实现 quantize(), export(), val() 等                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

| 目录 | 职责 |
|------|------|
| `src/model_optimizer/cli.py` | CLI 入口 |
| `src/model_optimizer/launcher.py` | 命令分发 |
| `src/model_optimizer/models/` | 模型抽象与具体实现 |
| `src/model_optimizer/quantization/` | 量化逻辑（PTQ、ONNX 量化等） |
| `src/model_optimizer/convert/` | 格式转换、ONNX 导出 |
| `src/model_optimizer/trt_build/` | ONNX → TensorRT 引擎 |
| `src/model_optimizer/calibrate/` | 校准数据收集 |
| `src/model_optimizer/evaluate/` | 评测与精度比较 |
| `src/model_optimizer/infer/` | PyTorch / TensorRT 推理 |
| `src/model_optimizer/webui/` | Gradio Web 界面 |
| `config/` | 量化与构建配置 |

---

## 3. 核心设计：模型注册与多态

### 3.1 模型注册表 (registry)

```python
# models/registry.py
ModelMaps = {}
register_model_cls("pi05_libero", Pi05Model)
register_model_cls("pi05_libero/vit", Vit)
register_model_cls("pi05_libero/llm", LLM)
register_model_cls("pi05_libero/expert", Expert)
register_model_cls("yolo", YoloModel)
```

- 子命令通过 `model_name` 查表，获取对应模型类
- 支持按子模型细分（如 `pi05_libero/llm`）

### 3.2 基础 Model 抽象

```python
# models/model.py
class Model:
    - get_calibrate_dataset(calib_data_file)
    - quantize(quant_cfg, calib_data, export_dir)  # 默认实现用 modelopt
    - val(dataset, batch_size, ...)                # 子类实现
    - export_onnx(...)                             # 子类实现
```

- 基类提供校准数据加载、默认量化流程（基于 modelopt）
- 子类覆盖 `val`、`export_onnx` 等接口

### 3.3 典型调用链（以量化为例）

```
quantize_cli()
  → model_cls = get_model_cls(model_name)     # 如 pi05_libero/llm
  → model = model_cls.construct_from_name_path(model_name, model_path)
  → model.quantize(quant_cfg, calib_data, export_dir)
```

- `construct_from_name_path` 由各模型类实现，内部完成加载和封装
- 后续量化、导出等都依赖模型的 `quantize` / `export` 等接口

---

## 4. 主要功能模块实现

### 4.1 量化 (quantize)

- 配置：通过 Python 文件（如 `config/quant/llm_nvfp4_quant_cfg.py`）提供 `QUANT_CFG`
- 依赖：NVIDIA ModelOpt (`modelopt.torch.quantization`)
- 流程：加载模型 → 校准 → 量化 → 导出 ONNX（部分模型带 NVFP4 后处理）
- 子模型实现：
  - **LLM**：使用 `quantize_model` 做 PTQ，再 `export` + `_nvfp4_post_processing`
  - **YOLO**：ONNX 导出后再做 ONNX 量化
  - **Expert / Vit**：各自的 `quantize` 实现

### 4.2 导出 (export)

- 入口：`convert_model()`，通过 `model_name` 查表，调用对应 `model.export(export_dir)`
- π0.5：拆成 vit / llm / expert 三个 ONNX，分别导出
- 配置驱动，支持不同精度（fp16、bf16 等）

### 4.3 校准 (calibrate)

- 目前以 YOLO 为主：用 `YOLOCalibCollector` 和 `YoLoCalibrationData` 做数据收集
- π0.5：在推理脚本中通过 `Pi05LLMCalibCollector`、`Pi05ExpertCalibCollector` 收集并保存 `.pt`

### 4.4 编译 (build)

- 使用 TensorRT 的 `trtexec` / Python API 将 ONNX 编译为引擎
- 配置：如 `config/build_configs/llm_build_cfg.py` 指定精度、shape、workspace 等

### 4.5 评测 (eval)

- 统一走 `model.val()`，内部按模型类型不同实现
- π0.5：用 `Pi05Metric` 等做 KV、hidden state 等对比

### 4.6 比较 (compare)

- 比较两套推理结果（如 PyTorch vs TensorRT）
- 通过 `compare_predictions` 做相似度、L1 等

---

## 5. 依赖与集成

### 5.1 外部依赖

| 依赖 | 用途 |
|------|------|
| `modelopt` (NVIDIA Model Optimizer) | PTQ、FP8/NVFP4、ONNX 量化 |
| `transformers` | HuggingFace 模型加载 |
| `openpi` | π0.5 策略与数据加载 |
| `ultralytics` | YOLO 模型与推理 |
| `tensorrt` | 构建和运行 TensorRT 引擎 |
| `gradio` | WebUI |

### 5.2 配置加载

```python
# config/config.py
load_settings(file_path)  # 使用 importlib 动态执行 Python 配置
```

- 配置为可执行 Python 文件，支持复杂逻辑和依赖

---

## 6. 数据流

```
原始 PyTorch 模型
       │
       ├─ calibrate ──→ 校准数据 (.pt / .npy)
       │
       ├─ quantize (PTQ) ──→ 量化后 PyTorch
       │
       └─ export ──→ ONNX（含 QDQ / NVFP4 等）
              │
              └─ build ──→ TensorRT 引擎 (.engine)
                     │
                     └─ infer (standalone_inference_script.py)
```

---

## 7. 总结

- **架构模式**：命令式 CLI + 模型注册表，通过 `model_name` 路由到不同实现
- **抽象设计**：Model 基类定义接口，各子模型实现 quantize / export / val
- **量化链路**：依赖 NVIDIA ModelOpt，支持 FP8、NVFP4、INT8 等
- **扩展性**：新增模型只需注册并实现接口，可覆盖多种架构（π0.5、YOLO 等）
- **双界面**：CLI 与 WebUI 共用底层逻辑，WebUI 用 Gradio 调用各模块
