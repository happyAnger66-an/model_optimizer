# Pi0.5 WebSocket 策略服务（`serve_policy.py`）

在 **model_optimizer** 仓库中，使用 `scripts/deployment/pi05/serve_policy.py` 启动与 openpi `serve_policy` 相同协议的 **WebSocket 策略服务**，并支持 **PyTorch** 或 **TensorRT** 推理；TensorRT 下各子图 engine 的指定方式与 [`standalone_inference_script.py`](../../scripts/deployment/pi05/standalone_inference_script.py) 一致。

## 依赖

- 已安装 **openpi**（含 `openpi.policies`、`openpi.policies.policy_config`、`openpi.serving.websocket_policy_server`、`openpi.training.config`）
- **`tyro`**、**`addict`**
- 可 `import model_optimizer`（安装本仓库或设置 `PYTHONPATH`）
- TensorRT 模式：GPU、与构建 engine 时一致的 TensorRT / CUDA 环境

## 运行位置

在 **model_optimizer 仓库根目录** 执行（保证相对路径与文档示例一致）：

```bash
cd /path/to/model_optimizer
```

## 常用参数说明

| 参数 | 说明 |
|------|------|
| **`--checkpoint-dir`** | 训练 checkpoint 目录（必填），内含权重等 |
| **`--config-name`** | `openpi.training.config.get_config` 注册名，或指向 `cfg = TrainConfig(...)` 的 `.py` 文件路径 |
| **`--inference-mode`** | `pytorch`（默认）或 `tensorrt` |
| **`--host` / `--port`** | 监听地址与端口（默认 `0.0.0.0:8000`） |
| **`--robot-type`** | 传入 `create_trained_policy(..., robot_type=...)` |
| **`--default-prompt`** | 观测中缺少 prompt/task 时的默认文本 |
| **`--record`** | 使用 `PolicyRecorder` 记录请求与响应 |
| **`--enable-value-endpoint`** | 是否允许 WebSocket `score` 请求（默认关闭；需策略实现 `score_observation`） |
| **`--precision`** | 仅 TRT：`fp16` / `bf16` / `fp32` |
| **`--trt-engine-path`** | TRT engine **目录** |
| **`--vit-engine` / `--llm-engine` / `--expert-engine` / `--denoise-engine` / `--embed-prefix-engine`** | 上述目录下的 engine **文件名**；按需只填已替换的子图 |

完整参数列表：

```bash
python scripts/deployment/pi05/serve_policy.py --help
```

## 示例：PyTorch（默认）

```bash
python scripts/deployment/pi05/serve_policy.py \
  --checkpoint-dir /path/to/pytorch_pi05_libero/ \
  --config-name pi05_libero \
  --inference-mode pytorch \
  --port 8000
```

可选：指定 `robot_type`、默认 prompt：

```bash
python scripts/deployment/pi05/serve_policy.py \
  --checkpoint-dir /path/to/pytorch_pi05_libero/ \
  --config-name pi05_libero \
  --robot-type unified_robot \
  --default-prompt "pick up the cube" \
  --inference-mode pytorch \
  --host 0.0.0.0 \
  --port 8000
```

使用自定义训练配置 **Python 文件**（文件中定义 `cfg` / `config` / `train_config` 为 `TrainConfig` 实例）：

```bash
python scripts/deployment/pi05/serve_policy.py \
  --checkpoint-dir /path/to/checkpoint/exp/30000 \
  --config-name /path/to/my_train_cfg.py \
  --inference-mode pytorch \
  --port 8000
```

## 示例：TensorRT（可替换各 engine）

`--trt-engine-path` 指向 **同一目录**，各 `--*-engine` 为该目录下的文件名；只需替换实际存在的 engine，未指定的子图仍走 PyTorch（若 executor 未 patch 对应模块）。

**全套子图示例**（与 `standalone_inference_script` 的 TRT 参数一致）：

```bash
python scripts/deployment/pi05/serve_policy.py \
  --checkpoint-dir /path/to/pytorch_pi05_libero/ \
  --config-name pi05_libero \
  --inference-mode tensorrt \
  --trt-engine-path /tmp/build/pi05/ \
  --vit-engine vit.engine \
  --llm-engine llm.engine \
  --expert-engine expert.engine \
  --denoise-engine denoise.engine \
  --embed-prefix-engine embed_prefix.engine \
  --precision bf16 \
  --port 8000
```

**仅替换部分 engine**（例如只换 LLM 与 expert）：

```bash
python scripts/deployment/pi05/serve_policy.py \
  --checkpoint-dir /path/to/pytorch_pi05_libero/ \
  --config-name pi05_libero \
  --inference-mode tensorrt \
  --trt-engine-path /tmp/build/pi05/ \
  --llm-engine llm_nvfp4.engine \
  --expert-engine expert_bf16.engine \
  --port 8000
```

## 协议与客户端

- 服务实现来自 **`openpi.serving.websocket_policy_server.WebsocketPolicyServer`**：首帧下发 **metadata**（msgpack），之后循环 **收观测 → `infer` / `score` → 回包**（含 `server_timing` 等字段）。
- 客户端需与 openpi 侧 **websocket client** 一致（例如 `openpi_client` 中的策略客户端）；具体字段与 `Policy.infer` 输入对齐，参见 openpi 文档与 `libero_policy` / `gr00t_policy` 等 transform 约定。

## 相关文档

- Pi0.5 总览与导出 / 量化：[`pi05.md`](./pi05.md)
- 单机推理与 TRT 说明：[`eval.md`](../eval.md) 中 TensorRT 与 `standalone_inference_script` 章节
