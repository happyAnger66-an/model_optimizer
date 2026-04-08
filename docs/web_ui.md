# LeRobot 评估 WebUI（`lerobot_eval_webui`）

浏览器实时查看 **π0.5（openpi）** 在 LeRobot 离线数据上的对齐推理：prompt、RGB、gt/pred 动作与滚动误差曲线。由 **WebSocket Server** 流式推送事件，**静态页面** 订阅展示。

更通用的离线评估与数据配置说明见 [eval.md](eval.md)；本文专注 WebUI 的启动方式、参数与标定数据采集。

---

## 架构概要

| 组件 | 说明 |
|------|------|
| **入口脚本** | `scripts/deployment/pi05/lerobot_eval_webui_server.py`（薄封装，实现位于同目录包 `lerobot_eval_webui/`） |
| **推理** | 独立 **守护线程** 同步执行：加载数据集与 `Policy`、按 chunk 调用 `policy.infer`；通过 `asyncio.run_coroutine_threadsafe` 把待推送消息交给主线程 |
| **WebSocket** | 主线程 **asyncio** 事件循环：`websockets` 监听、广播、处理 client 的 `pause`/`resume` |
| **Client** | `scripts/deployment/pi05/webui_client/`（`index.html` + 静态资源）；启动时 server 会写入 `server_hint.json` 供页面默认连接地址 |

这样可避免在 WS 回调里做重推理阻塞事件循环；GPU/CUDA 上下文固定在推理线程中。

---

## 环境与依赖

与 [eval.md](eval.md) 中「依赖与环境」一致，至少需要：

- `lerobot`（与 openpi 所用 API 兼容）、`tyro`、`websockets`、openpi 依赖；
- 可访问 openpi 源码，例如：

```bash
export PYTHONPATH="/path/to/model_optimizer/third_party/openpi/src:${PYTHONPATH}"
```

使用 **TensorRT** 推理时还需 model_optimizer 的 `src`（与 `lerobot_eval_compare.py` 相同）：

```bash
export PYTHONPATH="/path/to/model_optimizer/third_party/openpi/src:/path/to/model_optimizer/src:${PYTHONPATH}"
```

检查点目录需含 `model.safetensors` 与 `assets/<asset_id>/` 归一化统计（与 `create_trained_policy` 一致）。

---

## 快速启动 Server

```bash
export PYTHONPATH="third_party/openpi/src:src:${PYTHONPATH}"

python scripts/deployment/pi05/lerobot_eval_webui_server.py \
  --checkpoint /path/to/checkpoint \
  --config pi05_libero \
  --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero \
  --start-index 0 \
  --num-samples 500 \
  --host 0.0.0.0 \
  --port 8765
```

进程启动后 **立即** 监听 WebSocket；数据集与策略在推理线程中加载。先连接的 client 会先收到「加载中」的 `meta`，就绪后再收到完整 `meta` 与 `step` 流。

---

## 命令行参数说明

以下为 `lerobot_eval_webui.config.Args`（tyro）字段；命令行一般为 **kebab-case**（如 `--num-samples`）。

### 数据与策略（必选 / 常用）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `checkpoint` | （必填） | 检查点路径，含 `model.safetensors` 与 `assets/` |
| `config` | `pi05_libero` | `openpi.training.config` 中已注册的 `TrainConfig` 名称 |
| `num-samples` | `500` | 从 `start-index` 起连续评估的帧数（与数据集长度取 min） |
| `start-index` | `0` | 数据集起始全局下标 |
| `dataset-root` | `None` | LeRobot 本地根目录；省略则依赖 Hub/缓存行为（视 lerobot 版本而定） |
| `device` | `None` | 如 `cuda:0`；省略则由 openpi 策略侧默认选择 |

### 推理后端

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `inference-mode` | `pytorch` | `pytorch` 或 `tensorrt` |
| `precision` | `bf16` | TensorRT 路径精度：`fp16` / `bf16` / `fp32` |
| `engine-path` | `""` | TensorRT 引擎文件所在**目录**（`tensorrt` 模式必填） |
| `vit-engine` / `llm-engine` / `expert-engine` / `denoise-engine` / `embed-prefix-engine` | `""` | 各子引擎**文件名**（与 `engine-path` 拼接）；留空则该子模块仍走 PyTorch |

### WebSocket 与 Client 提示

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `host` | `0.0.0.0` | 监听地址 |
| `port` | `8765` | 监听端口 |
| `path` | `/ws` | WebSocket 路径；与 client 填写的 URL 路径一致 |
| `client-ws-url` | `None` | 写入 `webui_client/server_hint.json` 的默认 `ws://…`。**远端浏览器**访问时建议设为 server 的局域网地址，例如 `ws://192.168.1.10:8765/ws`。未设置且 `host` 为 `0.0.0.0` 或 `::` 时，hint 使用 `ws://127.0.0.1:{port}{path}` |

### 图像与推送节奏

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `send-wrist` | `False` | 是否在 `step` 事件中附带 `observation/wrist_image` 的 JPEG（默认仅 base 相机） |
| `jpeg-quality` | `85` | JPEG 质量 |
| `max-fps` | `0` | 限制推送 **step 事件** 速率（条/秒）；`0` 表示不限制 |
| `history-size` | `0` | 保留最近 N 条已推送消息；新连接会先收到当前 `meta` 再 **回放** 缓存（`0` 表示不缓存） |

### 误差指标口径（相对误差）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `rel-eps` | `1e-8` | 相对误差分母 eps：`rel = |pred-gt| / max(|gt|, rel_eps)`，用于屏蔽 `gt≈0` 导致的发散 |

### GPU 监控（可选）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `gpu-stats-interval-sec` | `1.0` | 周期推送 **``type=gpu_stats``**（需本机 `nvidia-smi`）。`0` 关闭。消息**不**进入 `history-size` 回放，避免刷屏 |
| `gpu-device-index` | `None` | `nvidia-smi -i` 的 GPU 编号；`None` 时从 `--device` 解析 `cuda:N`，否则为 `0` |

`gpu_stats` 含 `gpu_util_pct` / `mem_util_pct`（0–100，来自 `utilization.gpu` / `utilization.memory`）及 `device_index`。

### Pi0.5 标定数据采集（仅 PyTorch）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `calib-save-path` | `None` | 标定输出**根目录**。仅当 `inference-mode=pytorch` 时生效：在每次 `policy.infer` 上对 LLM / Expert / ViT 挂 forward hook，采集输入张量。**TensorRT 模式不支持**（会忽略并打日志）。 |

采集结束（或推理线程退出）时，在该目录下为各子模块写入：

- `{name}_calib_manifest.json`：分片清单与 `total_samples`（`name` 为 `pi05_llm`、`pi05_expert`、`pi05_vit`）
- `{name}_calib_shards/shard_XXXXX.pt`：每条分片内为若干条样本（采集侧按条数周期落盘，降低内存峰值）

**不再**默认合并为单个 `*_calib_datas.pt`；若目录里仍有历史合并文件，量化加载逻辑可兼容（见下文「标定数据用于量化」）。

---

## WebSocket 协议（v1）

消息体为 **JSON 文本**（UTF-8），字段随类型变化。

| `type` | 方向 | 说明 |
|--------|------|------|
| `meta` | server → client | 连接后：若 bundle 未就绪，先发 `phase=loading` 的短提示；加载完成后广播完整 `meta`（含 `run_id`、`repo_id`、`action_horizon`、`start_index`、`end_index_exclusive`、`backend`、`send_wrist`、`jpeg_quality`；若开启标定则含 `calib_save_path`） |
| `step` | server → client | 时间对齐后的单步：含 `episode_id`、`global_index`、`k_in_chunk`、`gt_action` / `pred_action`、`metrics`（如 mse/mae）、可选 `images`（base64 JPEG）、`prompt`（chunk 首步）等 |
| `done` | server → client | 本 run 在 `[start_index, end)` 内 step 已全部推送；随后 **server 关闭并结束整个进程** |
| `error` | server → client | 推理管线异常时的错误信息 |
| `control` | client → server | `{"type":"control","action":"pause"|"resume"}`：在 **下一个 chunk 的 `infer` 之前** 阻塞或恢复 |
| `control_ack` | server → client | 对控制的确认；连接成功后会发一条 `action=sync`，带当前是否暂停 |
| `gpu_stats` | server → client | 周期推送 GPU / 显存控制器利用率（见 `--gpu-stats-interval-sec`）；`run_id` 与当前 meta 一致 |

**时间对齐语义（与 compare 脚本一致）**

- 每隔 `action_horizon` 帧做一次 `infer`，得到 `pred[0..H-1]`，与 label `gt[0..H-1]` 按 `k` 对齐，`global_index = idx + k`。
- 若 chunk 跨越 **不同 episode**，该 chunk 跳过。
- 推送完毕后发 `done`，然后进程退出；需再次评估请 **重新启动** server。

---

## 打开 Client（浏览器）

静态资源目录：

```text
scripts/deployment/pi05/webui_client/
```

推荐在该目录起本地 HTTP 服务（避免 `file://` 下部分浏览器限制）：

```bash
cd scripts/deployment/pi05/webui_client
python -m http.server 8000
```

浏览器访问 `http://127.0.0.1:8000/`。页面会尝试读取同目录 **`server_hint.json`**（由 server 启动时生成）自动填入 WebSocket URL；也可手动改为 `ws://<host>:<port><path>`。

- 若在 **另一台机器** 上打开页面，请使用可路由到 server 的地址，并在启动 server 时设置 `--client-ws-url`，或在页面手动填写。
- 页面支持 **浅色/深色** 主题（偏好存 `localStorage`）。
- **暂停 / 继续**：通过上述 `control` 消息在下一 chunk 前生效；便于排查或截图。
- 若 server 侧开启 `--send-wrist`，可在 client 勾选显示 wrist 图。

---

## 标定数据用于量化

采集完成后，`calib-save-path` 指向的目录中包含各子模块的 **manifest + 分片**。在 model_optimizer 中对 Pi0.5 子模块量化时，可将 **`--calibrate_data` 设为该目录**（与传单个 `*_calib_datas.pt` 文件二选一；目录模式会按分片 **流式** 读取，避免一次性加载整表）。

各子模块使用的 `component` 名与 manifest 前缀一致：`pi05_llm`、`pi05_expert`、`pi05_vit`。具体 CLI 与配置见仓库内 `docs/pi05_quantize.md` 及 `model_optimizer calibrate` / `quantize` 相关说明。

---

## 代码模块布局（便于扩展）

实现位于 `scripts/deployment/pi05/lerobot_eval_webui/`：

| 模块 | 职责 |
|------|------|
| `config.py` | tyro `Args` |
| `protocol.py` | `StepEvent`、JSON 序列化、加载中 `meta` 常量 |
| `dataset.py` | LeRobot 数据集、repack、episode 列 |
| `bundle.py` | 加载数据集 + policy + TensorRT（可选）+ meta |
| `chunk_infer.py` | 单 chunk 推理与 step 消息生成 |
| `broadcaster.py` | WebSocket 客户端集合与历史回放 |
| `hints.py` | 写入 `webui_client/server_hint.json` |
| `gpu_stats.py` | `nvidia-smi` 采样 GPU 利用率 |
| `server.py` | `run_server` / `main`：WS 服务与推理线程 |
| `calib.py` | 启动 Pi0.5 calib collectors（供 bundle 使用） |

---

## 相关路径

| 路径 | 说明 |
|------|------|
| `scripts/deployment/pi05/lerobot_eval_webui_server.py` | CLI 入口 |
| `scripts/deployment/pi05/lerobot_eval_webui/` | Server 实现包 |
| `scripts/deployment/pi05/webui_client/` | 浏览器前端 |
| `src/model_optimizer/calibrate/collector/pi05.py` | 标定 hook 与分片写入 |
| `src/model_optimizer/calibrate/pi05_calib_load.py` | 标定目录 / 分片加载（量化侧使用） |
| [eval.md](eval.md) | `lerobot_eval_compare.py`、非 Libero 数据、TensorRT 对齐说明 |
