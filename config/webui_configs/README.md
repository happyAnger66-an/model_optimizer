# webui_configs —— lerobot_eval_webui_server 推理配置

`lerobot_eval_webui_server`（`scripts/deployment/pi05/lerobot_eval_webui_server.py`）的示例配置集合。
配置文件键为 `Args`（`lerobot_eval_webui/config.py`）的 **snake_case** 字段，支持 YAML / JSON。

## 用法

```bash
python scripts/deployment/pi05/lerobot_eval_webui_server.py \
  --webui-config config/webui_configs/<config>.yaml
```

- 命令行参数写在 `--webui-config` **之后**时可覆盖文件内的同名项，例如追加 `--port 9000`。
- 路径均为占位（`/srcs/openpi/pytorch_pi05_libero/`、`/tmp/pi05/build/quant`、`/tmp/pi05/onnx`），请按实际产物修改。

## 配置一览

| 文件 | 后端 / 模式 | 用途 | 前置产物 |
|---|---|---|---|
| `pytorch_baseline.yaml` | PyTorch 浮点 | 精度/对齐基线 | 仅需 checkpoint |
| `tensorrt_full.yaml` | TensorRT | 全引擎推理（vit/llm/expert/denoise） | 各 `.engine` |
| `tensorrt_flashrt_denoise.yaml` | TensorRT + FlashRT | vit/llm=TRT，denoise=仓内 FlashRT FVK 整循环 | TRT 引擎 + `build_flashrt_kernels.sh` 产物 |
| `tensorrt_flashrt_denoise_compare.yaml` | PyTorch vs TensorRT + FlashRT | PyTorch 参考路 vs vit/llm=TRT、denoise=FlashRT 的目标路 | TRT 引擎 + FlashRT kernel + 可选 act scales |
| `tensorrt_denoise_engine.yaml` | TensorRT | vit/llm/denoise 均 TRT engine（与 flashrt 版同 vit/llm 布局，denoise 用 `.engine`） | 含 `denoise_*.engine` 的 TRT 目录 |
| `tensorrt_native_denoise.yaml` | TensorRT + FlashRT | 同上（精简占位路径，与 flashrt 配置等价目标） | 同上 |
| `tensorrt_vit_batch.yaml` | TensorRT | 全引擎 + **多视角 batching**（延时优化） | vit 引擎须支持动态 batch |
| `pt_trt_compare.yaml` | PyTorch vs TensorRT | 双路逐 chunk 对比（含 ViT PT/TRT 对比） | TRT 各 `.engine` |
| `trt_trt_compare.yaml` | TensorRT vs TensorRT | 双引擎对比（如 FP8 vs NVFP4） | 两套 `.engine` |
| `onnxrt.yaml` | ONNX Runtime | ORT 后端推理 | `.onnx` 模型目录 |

## 关键字段

| 字段 | 说明 |
|---|---|
| `checkpoint` | PI0Pytorch 权重目录（必填） |
| `config` | 模型配置名（如 `pi05_libero`）。注意是 `config` 而非 `config_name` |
| `inference_mode` | `pytorch` / `tensorrt` / `onnxrt` |
| `engine_path` | TensorRT 引擎目录；`vit_engine`/`llm_engine`/… 为其下文件名 |
| `vit_batch_views` | 把所有相机视角堆成 batch 维**一次过 vit 引擎**（SigLIP 各图独立、数值等价） |
| `trt_vit_scale_fix` | TRT ViT 输出补乘 ``sqrt(hidden_size)``，修复部分 vit.engine 相对 PyTorch 精度偏低；等价 ``PI05_TRT_VIT_SCALE_FIX=1``（YAML 为 true 时优先） |
| `ort_engine_path` / `ort_*_engine` | ONNX Runtime 模型目录与文件名 |
| `noise` / `noise_seed` | `fixed` 时按 chunk 确定性生成推理初值，便于多后端复现对比 |

## 多视角 batching（`tensorrt_vit_batch.yaml`）注意

- **前置**：vit 引擎必须支持动态 batch —— 用 `vit.py`（已放开 `pixel_values` 的 batch 动态轴）导出，
  并用 `config/build_configs/vit_build_cfg.py`（`opt/max` batch = `_NUM_VIEWS` ≥ 实际视角数）编译。
  旧的 batch=1 静态引擎开 `vit_batch_views` 会维度报错。
- **互斥**：`vit_batch_views` 不能与 `embed_prefix_engine`（整图 embed_prefix 引擎）同用。
- 启动日志应出现 `embed_prefix: batched SigLIP (all views in one vit engine call)`；计时 hook 从
  `trt.vit.get_image_features`（N 次/infer）变为 `trt.embed_prefix_vit_batched`（1 次/infer）。

## 对比模式互斥

`compare_mode` / `ptq_compare` / `ptq_trt_compare` / `ort_compare` / `trt_ort_compare` / `trt_trt_compare`
**互斥**，请勿同时开启。其中：

- `compare_mode` / `ptq_compare` / `ptq_trt_compare` 仅支持 `inference_mode=pytorch`（或其语义要求的固定后端）。
- `trt_trt_compare` 必须 `inference_mode=tensorrt`，且需 `engine_path` 与 `trt_trt_second_engine_path`。
- `trt_ort_compare` 需同时 `engine_path`（TRT）与 `ort_engine_path`（ONNX）。

## 典型流程串联

1. 量化/导出/编译各子图（见各 `config/quant/*`、`config/build_configs/*`）。
2. 选一个 webui 配置，按实际路径改 `checkpoint` / `engine_path` / 各 `*_engine`。
3. 启动 server，浏览器连 `ws://<host>:<port><path>`（默认 `ws://127.0.0.1:8765/ws`）。
