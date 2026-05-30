# Pi0.5 部署与评估工具

本目录包含 π0.5 在 LeRobot 数据集上的**离线评估 WebUI**（client-server）及相关脚本。

| 入口 | 说明 |
|---|---|
| [`pi05/lerobot_eval_webui_server.py`](pi05/lerobot_eval_webui_server.py) | **评估 WebUI 服务端**（兼容入口，实现见 `lerobot_eval_webui/` 包） |
| [`pi05/webui_client/`](pi05/webui_client/) | 浏览器静态前端（订阅 WebSocket step 流） |
| [`pi05/standalone_inference_script.py`](pi05/standalone_inference_script.py) | 无 WebUI 的独立推理脚本 |
| [`pi05/lerobot_eval_compare.py`](pi05/lerobot_eval_compare.py) | 命令行对比评估（非 WebUI） |

下文以 **`lerobot_eval_webui_server.py`** 为主，按**功能大类 → 参数小类**说明全部 CLI（基于 `lerobot_eval_webui/config.py` 的 `Args`，由 **tyro** 解析）。

> **说明**：本工具**没有 argparse 子命令**（无 `serve` / `compare` 等子命令树），只有一组扁平参数；大类是逻辑分组，便于查阅。

---

## 1. 快速开始

### 1.1 启动服务端

```bash
cd scripts/deployment/pi05

# 最简：PyTorch 浮点推理 + 本地 WebSocket
python lerobot_eval_webui_server.py \
  --checkpoint /path/to/pi05_checkpoint \
  --config pi05_libero \
  --num-samples 100

# 使用 YAML 配置（命令行写在后面可覆盖文件中的项）
python lerobot_eval_webui_server.py --webui-config eval_webui.yaml

# TensorRT 单路
python lerobot_eval_webui_server.py \
  --checkpoint /path/to/checkpoint \
  --inference-mode tensorrt \
  --engine-path /path/to/trt_engines \
  --precision fp16
```

启动后会：

1. 监听 WebSocket（默认 `0.0.0.0:8765/ws`）
2. 将默认 client 地址写入 `webui_client/server_hint.json`
3. 后台线程跑数据集推理，按 step **流式推送**事件
4. 评估区间结束后发 `type=done` 并**退出进程**

### 1.2 打开浏览器客户端

```bash
# 任意静态 HTTP 服务即可（示例）
cd webui_client && python -m http.server 8080
# 浏览器打开 http://127.0.0.1:8080/ ，Connect 到 ws://127.0.0.1:8765/ws
```

远端访问时，启动 server 时指定：

```bash
--client-ws-url ws://<server_ip>:8765/ws
```

---

## 2. 配置文件

| 参数 | CLI 名 | 作用 |
|---|---|---|
| `webui_config` | `--webui-config` / `--webui_config` | 加载 `.yaml` / `.yml` / `.json`；键为 `Args` 字段名的 **snake_case** |

**优先级**：配置文件展开为默认 argv → **其后**的命令行参数覆盖文件。

**示例** `eval_webui.yaml`：

```yaml
checkpoint: /data/pi05_ckpt
config: pi05_libero
num_samples: 200
inference_mode: tensorrt
engine_path: /data/trt_fp16
host: 0.0.0.0
port: 8765
noise: fixed
noise_seed: 42
```

YAML 需安装 `PyYAML`（`pip install pyyaml`）。

---

## 3. CLI 参数总览（按大类）

### 3.1 模型与数据集

| 参数 | CLI | 类型 / 默认 | 作用 |
|---|---|---|---|
| `checkpoint` | `--checkpoint` | **必填** `Path` | π0.5 策略 checkpoint 目录 |
| `config` | `--config` | `str`，默认 `pi05_libero` | 训练/模型配置名（与 openpi 配置体系一致） |
| `num_samples` | `--num-samples` | `int`，默认 `500` | 评估样本数（按数据集 global index 推进） |
| `start_index` | `--start-index` | `int`，默认 `0` | 数据集起始 global index |
| `dataset_root` | `--dataset-root` | `Path \| None` | LeRobot 数据集根目录；`None` 时用配置内默认路径 |
| `device` | `--device` | `str \| None` | 设备，如 `cuda:0`；`None` 由策略自动选择 |
| `precision` | `--precision` | `fp16` / `bf16` / `fp32`，默认 `bf16` | PyTorch 推理精度（TensorRT 路仍由引擎决定） |

---

### 3.2 推理模式（单路）

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `inference_mode` | `--inference-mode` | `pytorch` | 单路后端：`pytorch` / `tensorrt` / `onnxrt` |

**约束**：

- 开启任意 **对比模式**（见 §3.3）时，**忽略** `inference_mode`（由对比标志决定双路组合）。
- `inference_mode=tensorrt` 时必须 `--engine-path`。
- `inference_mode=onnxrt` 时必须 `--ort-engine-path`。

---

### 3.3 对比模式（双路，互斥）

以下标志**至多开启一个**；同时开启多个会报错：

`compare_mode` / `ptq_compare` / `ptq_trt_compare` / `ort_compare` / `trt_ort_compare` / `trt_trt_compare`

| 大类 | 参数 | CLI | 双路含义 | 必要条件 |
|---|---|---|---|---|
| **PyTorch vs TensorRT** | `compare_mode` | `--compare-mode` | 浮点 PyTorch vs TRT 引擎 | `--engine-path` |
| **ViT 子图对比** | `vit_pt_trt_compare` | `--vit-pt-trt-compare` | 在 `compare_mode` 下额外对比 ViT：PT `get_image_features` vs TRT `vit.engine` | 需 `--compare-mode` |
| **PyTorch vs PTQ** | `ptq_compare` | `--ptq-compare` | 浮点 PyTorch vs 同权重 **选择性 PTQ**（fake quant） | `--ptq-quant-cfg`、`--ptq-calib-dir`、非空 `--ptq-parts`；且 `inference_mode=pytorch` |
| **PTQ vs TensorRT** | `ptq_trt_compare` | `--ptq-trt-compare` | PTQ PyTorch vs TRT 引擎 | 同上 PTQ 参数 + `--engine-path`；`inference_mode=pytorch` |
| **PyTorch vs ONNX Runtime** | `ort_compare` | `--ort-compare` | 浮点 PyTorch vs ORT | `--ort-engine-path` |
| **TensorRT vs ONNX Runtime** | `trt_ort_compare` | `--trt-ort-compare` | TRT vs ORT 端到端 action | `--engine-path` + `--ort-engine-path` |
| **TensorRT vs TensorRT** | `trt_trt_compare` | `--trt-trt-compare` | 两套 TRT 引擎目录对比 | `--inference-mode tensorrt`、`--engine-path`、`--trt-trt-second-engine-path` |

WebUI 的 `step` 事件中，第二路预测常见字段：

- `pred_action_trt` — TRT / ORT / 第二套 TRT
- `pred_action_ptq` — PTQ 路

协议细节见仓库内 `docs/quantize_ptq_compare.md`（若存在）。

---

### 3.4 TensorRT 引擎路径

用于 `inference_mode=tensorrt` 或各类 TRT 对比路。

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `engine_path` | `--engine-path` | `""` | TRT 引擎**根目录**（含各子图 `.engine`） |
| `vit_engine` | `--vit-engine` | `""` | ViT 引擎文件名；空则用目录内默认名 |
| `llm_engine` | `--llm-engine` | `""` | LLM 引擎文件名 |
| `expert_engine` | `--expert-engine` | `""` | Action expert 引擎文件名 |
| `denoise_engine` | `--denoise-engine` | `""` | Denoise 子图引擎文件名 |
| `embed_prefix_engine` | `--embed-prefix-engine` | `""` | Embed prefix 引擎文件名 |

---

### 3.5 ONNX Runtime 引擎路径

用于 `inference_mode=onnxrt` 或 `ort_compare` / `trt_ort_compare`。

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `ort_engine_path` | `--ort-engine-path` | `""` | ONNX 模型目录（`vit.onnx`、`llm.onnx` 等） |
| `ort_vit_engine` | `--ort-vit-engine` | `""` | ViT ONNX 文件名 |
| `ort_llm_engine` | `--ort-llm-engine` | `""` | LLM ONNX 文件名 |
| `ort_expert_engine` | `--ort-expert-engine` | `""` | Expert ONNX 文件名 |
| `ort_denoise_engine` | `--ort-denoise-engine` | `""` | Denoise ONNX 文件名 |
| `ort_embed_prefix_engine` | `--ort-embed-prefix-engine` | `""` | Embed prefix ONNX 文件名 |
| `ort_providers` | `--ort-providers` | 见下 | ORT Execution Provider 顺序（仅保留环境可用的） |

默认 `ort_providers`：

`TensorRTExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`

NVFP4 等常需 `TensorRTExecutionProvider`；纯 CUDA 调试可设：

`--ort-providers CUDAExecutionProvider CPUExecutionProvider`

---

### 3.6 双 TensorRT 对比（第二路引擎）

仅在 `--trt-trt-compare` 时使用。

| 参数 | CLI | 作用 |
|---|---|---|
| `trt_trt_second_engine_path` | `--trt-trt-second-engine-path` | 第二套 TRT 引擎根目录（**必填**） |
| `trt_trt_second_vit_engine` | `--trt-trt-second-vit-engine` | 第二路 ViT 文件名；空则沿用主路 `--vit-engine` 规则 |
| `trt_trt_second_llm_engine` | `--trt-trt-second-llm-engine` | 第二路 LLM |
| `trt_trt_second_expert_engine` | `--trt-trt-second-expert-engine` | 第二路 expert |
| `trt_trt_second_denoise_engine` | `--trt-trt-second-denoise-engine` | 第二路 denoise |
| `trt_trt_second_embed_prefix_engine` | `--trt-trt-second-embed-prefix-engine` | 第二路 embed_prefix |

---

### 3.7 Polygraphy 子图对比（仅 `trt_ort_compare`）

在**加载阶段**对成对 ONNX / TRT 子图跑 Polygraphy `Comparator`，摘要写入 meta 的 `trt_ort_polygraphy`（非 WebUI 逐步对比）。

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `trt_ort_polygraphy_compare` | `--trt-ort-polygraphy-compare` | `false` | 开启 Polygraphy 对比 |
| `trt_ort_polygraphy_mark_all` | `--trt-ort-polygraphy-mark-all` | `false` | 用 `MARK_ALL` 暴露中间张量；须与 `rebuild_trt` 同开 |
| `trt_ort_polygraphy_rebuild_trt` | `--trt-ort-polygraphy-rebuild-trt` | `false` | 从 MARK_ALL 后 ONNX **现场编译** TRT，不用预置 `.engine` |
| `trt_ort_polygraphy_parts` | `--trt-ort-polygraphy-parts` | 空 | 仅跑列出的子图：`vit` / `embed_prefix` / `llm` / `expert` / `denoise` |
| `trt_ort_polygraphy_ort_providers` | `--trt-ort-polygraphy-ort-providers` | CUDA+CPU | Polygraphy 侧 ORT providers |
| `trt_ort_polygraphy_max_report_tensors` | `--trt-ort-polygraphy-max-report-tensors` | `256` | 每子图写入 meta 的张量条数上限（按 max_abs 排序） |
| `trt_ort_polygraphy_seed` | `--trt-ort-polygraphy-seed` | `0` | 合成输入的 numpy 随机种子 |

依赖：`polygraphy`、`onnx`、`onnxruntime-gpu` 等需自行安装。

---

### 3.8 PTQ 对比与层误差报告

用于 `ptq_compare` / `ptq_trt_compare`。

| 参数 | CLI | 作用 |
|---|---|---|
| `ptq_quant_cfg` | `--ptq-quant-cfg` | ModelOpt 量化配置：`.json` 或定义 `QUANT_CFG` 的 `.py` |
| `ptq_calib_dir` | `--ptq-calib-dir` | 校准数据目录（含 `pi05_{vit,llm,expert,denoise}_calib_*`） |
| `ptq_parts` | `--ptq-parts` | 要量化的子系统，可重复：`vit` `llm` `expert` `denoise` |
| `ptq_measure_quant_error` | `--ptq-measure-quant-error` | PTQ 后用校准数据再跑一遍，打印张量级 QDQ 误差 |
| `ptq_layer_report_path` | `--ptq-layer-report-path` | 将各 QuantLinear 相对 FP 的误差写入 JSON |
| `ptq_layer_report_samples` | `--ptq-layer-report-samples` | layer report 使用的连续样本数（默认 32，自 `start_index` 起） |
| `ptq_layer_report_histogram` | `--ptq-layer-report-histogram` | 是否附带 FP 激活 subsample 直方图 |
| `ptq_layer_report_hist_bins` | `--ptq-layer-report-hist-bins` | 直方图 bin 数（默认 40） |
| `ptq_layer_report_hist_max_elems` | `--ptq-layer-report-hist-max-elems` | 每层参与直方图的最大元素数（默认 100000） |

---

### 3.9 WebSocket 服务与客户端

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `host` | `--host` | `0.0.0.0` | 监听地址 |
| `port` | `--port` | `8765` | 监听端口 |
| `path` | `--path` | `/ws` | WebSocket 路径（`/ws` 与 `/ws/` 等价） |
| `client_ws_url` | `--client-ws-url` | `None` | 写入 `webui_client/server_hint.json` 的默认 URL；不设则 `host` 为 `0.0.0.0`/`::` 时用 `ws://127.0.0.1:port/path` |

**协议 v1（server → client）**

| `type` | 含义 |
|---|---|
| `meta` | 元数据；连接后先发 loading meta，加载完成后发完整 meta |
| `step` | 单步事件：GT/pred action、误差、prompt、JPEG 图像等 |
| `done` | 本 run 评估区间结束；随后 server 关闭连接并退出 |
| `log` | 可选日志 |
| `gpu_stats` | GPU 利用率（见 §3.12） |
| `control_ack` | 对 client 控制命令的确认 |

**协议 v1（client → server）**

```json
{"type":"control","action":"pause"}
{"type":"control","action":"resume"}
```

在**下一个 chunk 推理前**生效；server 回复 `control_ack`。

---

### 3.10 可视化与推送

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `send_wrist` | `--send-wrist` | `false` | 除 base 相机外是否推送 wrist 图像 |
| `jpeg_quality` | `--jpeg-quality` | `85` | step 中 JPEG 质量 |
| `max_fps` | `--max-fps` | `0` | 限制 step 推送帧率（events/s）；`0` 不限制 |
| `outbound_queue_maxsize` | `--outbound-queue-maxsize` | `0` | 推理线程 → WebSocket 的 Janus 队列容量；`0` 无界；正整数可反压推理 |
| `history_size` | `--history-size` | `0` | 缓存最近 N 条消息供新 client 回放；`0` 不缓存 |
| `rel_eps` | `--rel-eps` | `1e-8` | 相对误差分母：`rel = |pred-gt| / max(|gt|, rel_eps)` |

---

### 3.11 校准数据收集（PTQ 前置）

仅在 `inference_mode=pytorch` 或 `compare_mode` / `ptq_compare` 时生效；**TensorRT 单路不支持**。

| 参数 | CLI | 作用 |
|---|---|---|
| `calib_save_path` | `--calib-save-path` | 输出目录；每次 `policy.infer` 挂 hook，结束时写 `*_calib_manifest.json` 与 `*_calib_shards/` |
| `calib_max_samples` | `--calib-max-samples` | 每 component 收集上限；`0` 不限制；`1` 仅 1 条 |
| `calib_item` | `--calib-item` | `all` / `vit` / `llm` / `expert` / `denoise` / `embed_prefix` |

与 `standalone_inference_script.py --calib-save-path` 语义一致；量化时 `--calibrate_data` 指向该目录。

---

### 3.12 GPU 监控

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `gpu_stats_interval_sec` | `--gpu-stats-interval-sec` | `1.0` | 周期推送 `type=gpu_stats`（需本机 `nvidia-smi`）；`≤0` 关闭 |
| `gpu_device_index` | `--gpu-device-index` | `None` | `nvidia-smi -i` 的 GPU 下标；`None` 时从 `--device` 解析 `cuda:N`，否则 `0` |

---

### 3.13 流匹配初值（可复现）

| 参数 | CLI | 默认 | 作用 |
|---|---|---|---|
| `noise` | `--noise` | `random` | `random`：模型内随机采样；`fixed`：每数据 chunk 用确定性高斯 `noise` 传入 `Policy.infer` |
| `noise_seed` | `--noise-seed` | `0` | 仅 `noise=fixed` 时与 chunk 起点 index 一起决定初值（`numpy.random.SeedSequence`） |

双路对比时**两路共用同一块 noise**，便于公平对比。

---

## 4. 常用场景示例

### 4.1 PyTorch 浮点 + WebUI（默认）

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --dataset-root /data/lerobot/libero \
  --num-samples 200 \
  --start-index 0 \
  --device cuda:0
```

### 4.2 PyTorch vs TensorRT 对比

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --compare-mode \
  --engine-path /data/trt_fp16 \
  --vit-pt-trt-compare \
  --num-samples 100
```

### 4.3 浮点 vs 选择性 PTQ

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --ptq-compare \
  --ptq-quant-cfg /path/to/nvfp4.json \
  --ptq-calib-dir /path/to/calib_out \
  --ptq-parts llm expert denoise \
  --num-samples 50
```

### 4.4 TensorRT vs ONNX Runtime + Polygraphy

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --trt-ort-compare \
  --engine-path /data/trt_engines \
  --ort-engine-path /data/onnx_models \
  --trt-ort-polygraphy-compare \
  --trt-ort-polygraphy-parts vit llm expert
```

### 4.5 收集校准数据

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --calib-save-path /data/pi05_calib_run \
  --calib-item all \
  --calib-max-samples 32 \
  --num-samples 32
```

### 4.6 可复现推理（固定 noise）

```bash
python lerobot_eval_webui_server.py \
  --checkpoint /data/pi05 \
  --noise fixed \
  --noise-seed 42
```

---

## 5. 架构简述

```text
lerobot_eval_webui_server.py  →  lerobot_eval_webui.main()
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
              server.run_server()              eval_session.run_infer_worker()
              (asyncio WebSocket)                (thread: 数据集 + Policy.infer)
                    │                                   │
                    │         Janus.Queue (outbound)      │
                    └──────── outbound_bridge ──────────┘
```

- **训练/推理路径**：`bundle` 加载 checkpoint → `chunk_infer` 按 chunk 调用 `infer_backends` 策略 → 对齐 action → 编码图像 → 推送 `step`。
- **控制面**：浏览器经 WebSocket 发 `pause`/`resume`；推理线程在 chunk 边界检查 `infer_paused` Event。

---

## 6. 依赖与环境

| 组件 | 用途 |
|---|---|
| `websockets` | WebSocket 服务端 |
| `janus` | 线程 ↔ asyncio 队列桥接 |
| `tyro` | CLI 解析 |
| `PyYAML` | 可选，仅 `--webui-config` 读 YAML |
| LeRobot / openpi / model_optimizer | 数据集、Policy、PTQ |
| TensorRT / ONNX Runtime / Polygraphy | 对应推理与对比模式 |

需在能加载 π0.5 checkpoint 与 LeRobot 数据集的环境中运行（通常与 model_optimizer 主环境一致）。

---

## 7. 相关文档

| 文档 | 内容 |
|---|---|
| [`pi05/lerobot_eval_webui_server.py`](pi05/lerobot_eval_webui_server.py) 文件头 | 协议 v1、配置文件、noise、Polygraphy 摘要 |
| [`docs/quantize_ptq_compare.md`](../../docs/quantize_ptq_compare.md) | PTQ 对比协议字段（若已添加） |
| [`docs/optimizer/roadmap.md`](../../docs/optimizer/roadmap.md) | Thor 部署优化路线图 |

---

## 8. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-05-30 | 初版：基于 `lerobot_eval_webui/config.py` 整理全部 CLI 大类与小类 |
