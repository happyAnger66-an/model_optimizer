# Infer Server — Pi0.5 推理服务库

`model_optimizer.infer.server` 提供 Pi0.5 模型的推理服务能力，支持 **离线数据集推理**、**WebSocket 流式推送** 和 **在线策略服务** 三种使用方式。

## 目录

- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [运行模式](#运行模式)
- [核心 API](#核心-api)
  - [InferServer](#inferserver)
  - [WebSocketInferServer](#websocketinferserver)
  - [PolicyServer](#policyserver)
- [数据结构](#数据结构)
  - [ChunkPayload](#chunkpayload)
  - [StepResult](#stepresult)
  - [InferResult](#inferresult)
  - [PredictionPack](#predictionpack)
- [推理与后处理解耦](#推理与后处理解耦)
- [WebSocket 协议](#websocket-协议)
  - [离线推理 WebSocket](#离线推理-websocket)
  - [PolicyServer WebSocket（openpi_client 协议）](#policyserver-websocketopenpi_client-协议)
- [校准数据收集](#校准数据收集)
- [使用示例](#使用示例)

---

## 架构概览

### 离线推理（InferServer / WebSocketInferServer）

```
┌────────────────────────────────────────────────────────────┐
│                      ServerConfig (JSON)                   │
└────────────────────┬───────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │     InferServer     │  ← Facade
          │  (load / infer_chunk│
          │   / run_all / close)│
          └──┬─────────────┬────┘
             │             │
    ┌────────▼───┐   ┌─────▼──────────┐
    │ InferBackend│   │ ResultWorker   │  ← 可选后处理线程
    │ (Strategy)  │   │ (Consumer)     │
    └──┬─────────┘   └────────────────┘
       │
       ├─ SinglePyTorchBackend      (mode=pytorch)
       ├─ SingleTensorRTBackend     (mode=tensorrt)
       ├─ PtTrtCompareBackend       (mode=pt_trt_compare)
       ├─ PtPtqCompareBackend       (mode=pt_ptq_compare)
       └─ PtqTrtCompareBackend      (mode=ptq_trt_compare)
```

### 在线策略服务（PolicyServer）

```
record_unified.py (WebsocketClientPolicy)
        │  msgpack over WebSocket
        ▼
┌─ PolicyServer ──────────────────────────────────┐
│   WebSocket handler (兼容 openpi_client 协议)    │
│        │                                        │
│   ┌────▼─────────────────────────┐              │
│   │    PolicyBackend (Strategy)  │              │
│   │    ├─ LocalPolicyBackend     │ ← 本地 GPU   │
│   │    └─ RemotePolicyBackend    │ ← 转发远程   │
│   └──────────────────────────────┘              │
└─────────────────────────────────────────────────┘
```

**设计模式**：

| 模式 | 用途 |
|------|------|
| **Strategy** | `InferBackend` 5 种子类按 `mode` 自动选择；`PolicyBackend` 按 `serve.backend` 选择本地/远程 |
| **Facade** | `InferServer` 统一离线推理入口；`PolicyServer` 统一在线策略服务入口 |
| **Producer-Consumer** | 推理线程产出 `ChunkPayload`，`ResultWorker` 异步消费构建 `StepResult` |

---

## 快速开始

### 安装依赖

```bash
pip install model_optimizer
# WebSocket 模式额外需要：
pip install websockets janus
```

### 离线数据集推理

```python
from model_optimizer.infer.server import InferServer, load_config

config = load_config("config.json")
server = InferServer(config)
server.load()

result = server.run_all(on_step=lambda s: print(f"step {s.global_index}: MSE={s.metrics['mse']:.6f}"))
print(f"共 {len(result.steps)} 个 step")

server.close()
```

### 在线策略服务

```python
from model_optimizer.infer.server import PolicyServer, load_config

config = load_config("serve_config.json")
server = PolicyServer(config)
server.load()
server.run()  # 阻塞，等待客户端连接
```

客户端（`record_unified.py`）可零改动直接连接：

```python
from openpi_client import WebsocketClientPolicy
policy = WebsocketClientPolicy(host="localhost", port=8000)
result = policy.infer(obs)
```

---

## 配置参考

配置文件为 JSON 格式，对应 `ServerConfig` dataclass。

### 完整配置示例

```json
{
  "checkpoint": "/path/to/pi05/checkpoint",
  "config_name": "pi05_libero",
  "mode": "pytorch",
  "device": "cuda",
  "precision": "bf16",
  "rel_eps": 1e-8,
  "enable_result": true,

  "dataset": {
    "repo_id": null,
    "root": null,
    "num_samples": 500,
    "start_index": 0
  },

  "tensorrt": {
    "engine_path": "",
    "vit_engine": "",
    "llm_engine": "",
    "expert_engine": "",
    "denoise_engine": "",
    "embed_prefix_engine": ""
  },

  "ptq": {
    "quant_cfg": null,
    "calib_dir": null,
    "parts": []
  },

  "websocket": {
    "enabled": false,
    "host": "0.0.0.0",
    "port": 8765,
    "path": "/ws",
    "max_fps": 0,
    "history_size": 0,
    "gpu_stats_interval": 1.0,
    "jpeg_quality": 85,
    "send_wrist": false,
    "client_ws_url": null,
    "outbound_queue_maxsize": 0
  },

  "serve": {
    "host": "0.0.0.0",
    "port": 8000,
    "backend": "local",
    "remote_host": "localhost",
    "remote_port": 8000,
    "default_prompt": null,
    "robot_type": "unified_robot",
    "unify_action_mode": true,
    "enable_score": false,
    "value_temperature": 1.0,
    "record": false,
    "record_dir": "policy_records"
  },

  "calib": {
    "save_path": null,
    "max_samples": 0,
    "item": "all"
  }
}
```

### 字段说明

#### 顶层字段 (`ServerConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint` | `str` | `""` | **必填** Pi0.5 checkpoint 路径 |
| `config_name` | `str` | `"pi05_libero"` | openpi 训练配置名称 |
| `mode` | `InferMode` | `"pytorch"` | 运行模式，见[运行模式](#运行模式) |
| `device` | `str \| null` | `null` | PyTorch device（`"cuda"`, `"cuda:0"` 等） |
| `precision` | `str` | `"bf16"` | 浮点精度：`"fp16"` / `"bf16"` / `"fp32"` |
| `rel_eps` | `float` | `1e-8` | 相对误差计算的分母 epsilon |
| `enable_result` | `bool` | `true` | 是否启用后处理。`false` 时不启动后处理线程、不计算 metrics、不编码图像，推理路径零额外开销 |

#### `dataset` (`DatasetConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repo_id` | `str \| null` | `null` | LeRobot 数据集 repo_id（由训练配置自动确定） |
| `root` | `str \| null` | `null` | 本地数据集根目录（覆盖 HF 缓存路径） |
| `num_samples` | `int` | `500` | 推理样本数量 |
| `start_index` | `int` | `0` | 数据集起始索引 |

#### `tensorrt` (`TensorRTConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `engine_path` | `str` | `""` | TensorRT engine 根路径 |
| `vit_engine` | `str` | `""` | ViT engine 文件名 |
| `llm_engine` | `str` | `""` | LLM engine 文件名 |
| `expert_engine` | `str` | `""` | Expert engine 文件名 |
| `denoise_engine` | `str` | `""` | Denoise engine 文件名 |
| `embed_prefix_engine` | `str` | `""` | Embed prefix engine 文件名 |

#### `ptq` (`PTQConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `quant_cfg` | `str \| null` | `null` | 量化配置文件路径（`.json` 或 `.py`） |
| `calib_dir` | `str \| null` | `null` | 校准数据目录 |
| `parts` | `list[str]` | `[]` | 量化子模块：`"vit"` / `"llm"` / `"expert"` / `"denoise"` 的子集 |

#### `websocket` (`WebSocketConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 是否启用 WebSocket |
| `host` | `str` | `"0.0.0.0"` | 监听地址 |
| `port` | `int` | `8765` | 监听端口 |
| `path` | `str` | `"/ws"` | WebSocket 路径 |
| `max_fps` | `float` | `0` | 推送帧率限制（0 = 不限） |
| `history_size` | `int` | `0` | 新连接回放历史消息数（0 = 不回放） |
| `gpu_stats_interval` | `float` | `1.0` | GPU 统计推送间隔秒数（0 = 禁用） |
| `jpeg_quality` | `int` | `85` | 图像 JPEG 编码质量 (1-100) |
| `send_wrist` | `bool` | `false` | 是否同时推送腕部相机图像 |
| `client_ws_url` | `str \| null` | `null` | 前端 WebSocket 连接地址（meta 中下发） |
| `outbound_queue_maxsize` | `int` | `0` | 出站消息队列上限（0 = 无限） |

#### `serve` (`ServeConfig`)

PolicyServer 在线策略推理服务配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | `str` | `"0.0.0.0"` | WebSocket 监听地址 |
| `port` | `int` | `8000` | WebSocket 监听端口 |
| `backend` | `str` | `"local"` | 推理后端：`"local"`（本地 GPU）或 `"remote"`（转发远程） |
| `remote_host` | `str` | `"localhost"` | remote 模式下目标 serve_policy 主机 |
| `remote_port` | `int` | `8000` | remote 模式下目标 serve_policy 端口 |
| `default_prompt` | `str \| null` | `null` | 默认自然语言 prompt（传给 `create_trained_policy`） |
| `robot_type` | `str` | `"unified_robot"` | 机器人类型（影响动作空间映射） |
| `unify_action_mode` | `bool` | `true` | 是否启用统一动作空间（16 维） |
| `enable_score` | `bool` | `false` | 是否启用 score endpoint（需模型包含 RL value head） |
| `value_temperature` | `float` | `1.0` | score 的 temperature 参数 |
| `record` | `bool` | `false` | 是否记录推理请求（调试用，依赖 `openpi.PolicyRecorder`） |
| `record_dir` | `str` | `"policy_records"` | 推理记录保存目录 |

#### `calib` (`CalibConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_path` | `str \| null` | `null` | 校准数据保存路径（`null` = 不收集） |
| `max_samples` | `int` | `0` | 最大校准样本数（0 = 不限） |
| `item` | `str` | `"all"` | 收集目标：`"all"` / `"vit"` / `"llm"` / `"expert"` / `"denoise"` |

---

## 运行模式

通过 `mode` 字段选择，共 5 种：

| Mode | 说明 | 需要 `tensorrt` 配置 | 需要 `ptq` 配置 |
|------|------|:---:|:---:|
| `pytorch` | 单路 PyTorch 浮点推理 | - | - |
| `tensorrt` | 单路 TensorRT 引擎推理 | Y | - |
| `pt_trt_compare` | PyTorch + TensorRT 双路对比 | Y | - |
| `pt_ptq_compare` | PyTorch + PTQ fake-quant 双路对比 | - | Y |
| `ptq_trt_compare` | PTQ + TensorRT 双路对比 | Y | Y |

双路模式会加载两份策略，分别执行推理后计算配对误差指标。

---

## 核心 API

### `load_config`

```python
def load_config(path: str | Path) -> ServerConfig
```

从 JSON 文件加载配置，自动递归构建嵌套 dataclass 并执行校验。

**异常**：
- `FileNotFoundError` — 文件不存在
- `ValueError` — 模式所需字段缺失（如 `tensorrt` 模式未设 `engine_path`）

---

### InferServer

推理服务核心 Facade，线程安全。

```python
class InferServer:
    def __init__(self, config: ServerConfig) -> None: ...
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `config` | `ServerConfig` | 当前配置 |
| `meta` | `dict[str, Any]` | 运行元信息（加载后可用） |
| `is_loaded` | `bool` | 是否已加载 |
| `start_index` | `int` | 数据集起始索引 |
| `end` | `int` | 数据集结束索引（不含） |
| `action_horizon` | `int` | 动作时间步长 |

#### `load`

```python
def load(self, on_progress: Callable[[str, str], None] | None = None) -> None
```

加载模型、数据集，构建推理后端。

- `on_progress(stage, message)` — 可选进度回调，`stage` 为阶段标识（`"config"` / `"policy_pt"` / `"dataset"` / `"ready"` 等），`message` 为人类可读描述。
- 加载顺序：策略 → 数据集 → 校准收集器（可选）→ 推理后端 → meta
- **异常**：`RuntimeError`（重复调用）

#### `infer_chunk`

```python
def infer_chunk(self, idx: int) -> ChunkPayload | None
```

对单个数据索引执行纯推理，返回轻量 `ChunkPayload`。

**只做 `backend.predict()`**，不计算 metrics、不编码图像。如果 `idx` 不在有效对齐位置或跨越 episode 边界，返回 `None`。

- `idx` 必须满足 `(idx - start_index) % action_horizon == 0`
- `idx + action_horizon` 不能超过数据集末尾
- 同一 chunk 内所有帧必须属于同一 episode

#### `run_all`

```python
def run_all(
    self,
    on_step: Callable[[StepResult], None] | None = None,
) -> InferResult | None
```

遍历 `[start_index, end)` 内所有有效 chunk 执行推理。

- **`enable_result=True`**（默认）：启动后处理线程，异步构建 `StepResult`。`on_step` 回调在后处理线程中触发。返回 `InferResult`。
- **`enable_result=False`**：不启动后处理，不计算 metrics，不编码图像，返回 `None`。适用于只关心推理吞吐或校准数据收集。

#### `close`

```python
def close(self) -> None
```

释放资源（校准收集器、后端、数据集）。调用后可重新 `load()`。

---

### WebSocketInferServer

基于 WebSocket 的流式推理服务，内部持有一个 `InferServer` 实例。

```python
class WebSocketInferServer:
    def __init__(self, config: ServerConfig) -> None: ...
```

#### `run`

```python
def run(self) -> None
```

**同步阻塞入口**：启动 asyncio 事件循环，内部创建 WebSocket 服务器 + 推理线程。推理完毕后自动关闭服务器。

#### `start`

```python
async def start(self) -> None
```

**异步入口**：启动 WebSocket 服务、推理线程与 GPU 统计推送。适用于需要自行管理事件循环的场景。

---

### PolicyServer

在线策略推理 WebSocket 服务，兼容 openpi_client 协议（msgpack 序列化）。`record_unified.py` 的 `WebsocketClientPolicy` 可零改动直接连接。

```python
class PolicyServer:
    def __init__(self, config: ServerConfig) -> None: ...
```

支持两种推理后端（通过 `config.serve.backend` 选择）：

| 后端 | 说明 |
|------|------|
| `local` | 在本地 GPU 加载模型，直接调 `policy.infer(obs)` |
| `remote` | 转发请求到远程 openpi `serve_policy` 服务（代理模式） |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_loaded` | `bool` | 后端是否已加载 |

#### `load`

```python
def load(self, on_progress: Callable[[str, str], None] | None = None) -> None
```

加载策略后端。

- **local 模式**：加载模型到 GPU，复用 `policy_loader.load_policy_for_serve()`。根据 `config.mode` 支持 pytorch / tensorrt / ptq 等所有后端模式。传递 `serve.default_prompt`、`serve.robot_type`、`serve.unify_action_mode` 参数给 `create_trained_policy`。
- **remote 模式**：创建 `WebsocketClientPolicy` 连接远程 serve_policy 服务。
- `on_progress(stage, message)` — 可选进度回调。
- **异常**：`RuntimeError`（重复调用未先 `close()`）、`ValueError`（未知 backend 类型）。

#### `run`

```python
def run(self) -> None
```

**同步阻塞入口**：启动 WebSocket 服务，内部调 `asyncio.run(self.start())`。

#### `start`

```python
async def start(self) -> None
```

**异步入口**：启动 WebSocket 服务并等待关闭信号。适用于需要自行管理事件循环的场景。

内部行为：
1. 连接建立时，发送策略 metadata（msgpack 编码）。
2. 循环接收 obs（msgpack）→ 解析 `_request_type`（`"infer"` 或 `"score"`）→ 调用后端推理 → 附加 `server_timing` → 发送结果（msgpack）。
3. 提供 `/healthz` 健康检查端点（HTTP GET → 200 OK）。

#### `close`

```python
def close(self) -> None
```

释放后端资源。调用后可重新 `load()`。

#### PolicyBackend 接口

`PolicyServer` 内部通过 `PolicyBackend` 策略模式选择推理后端：

```python
class PolicyBackend(ABC):
    def infer(self, obs: dict) -> dict: ...      # 执行推理
    def score(self, obs: dict, *, value_temperature: float = 1.0) -> dict: ...  # 计算 value score（可选）
    @property
    def supports_score(self) -> bool: ...         # 是否支持 score
    @property
    def metadata(self) -> dict: ...               # 策略元信息
    def close(self) -> None: ...                  # 释放资源
```

| 实现 | 说明 |
|------|------|
| `LocalPolicyBackend` | 直接调 openpi `Policy.infer(obs)`。可选 `PolicyRecorder` 包装（`serve.record=true`）。`supports_score` 取决于模型是否有 `score_observation` 方法（RL value head）。 |
| `RemotePolicyBackend` | 持有 `openpi_client.WebsocketClientPolicy`，转发 `infer`/`score` 请求到远程 serve_policy 服务。`supports_score` 由远程服务 metadata 决定。 |

---

## 数据结构

### ChunkPayload

推理原始产物 — 轻量数据拷贝，**不含 metrics 和图像编码**。由 `infer_chunk` 直接返回。

```python
@dataclass
class ChunkPayload:
    idx: int                                   # 数据集索引
    episode_id: int                            # 所属 episode ID
    action_horizon: int                        # 动作时间步长
    pack: PredictionPack                       # 推理结果（pred/gt/timing）
    prompt: str | None = None                  # 自然语言 prompt
    raw_images: dict[str, np.ndarray] | None = None  # 观测图像原始 numpy
```

### StepResult

单个 step 的完整后处理结果（由 `ResultWorker` 异步构建）。

```python
@dataclass
class StepResult:
    episode_id: int                            # 所属 episode ID
    global_index: int                          # 全局 step 索引
    k_in_chunk: int                            # chunk 内偏移 (0 ~ action_horizon-1)
    is_chunk_start: bool                       # 是否为 chunk 首 step
    action_horizon: int                        # 动作时间步长
    gt_action: np.ndarray                      # 标注动作
    pred_action: np.ndarray                    # 主路预测动作
    metrics: dict[str, Any]                    # 误差指标
    pred_action_trt: np.ndarray | None = None  # TRT 路预测（双路模式）
    pred_action_ptq: np.ndarray | None = None  # PTQ 路预测（双路模式）
    prompt: str | None = None                  # prompt（仅 chunk 首 step）
    images: dict[str, str] | None = None       # JPEG base64 图像（仅 chunk 首 step）
    timing: dict[str, float] | None = None     # 推理耗时（仅 chunk 首 step）
```

#### `metrics` 字段内容

| 键 | 说明 | 始终存在 |
|----|------|:---:|
| `mse` | 均方误差 | Y |
| `mae` | 平均绝对误差 | Y |
| `mse_pt` / `mae_pt` | 同上（别名，兼容协议） | Y |
| `mae_per_dim` | 逐维绝对误差列表 | Y |
| `mse_per_dim` | 逐维均方误差列表 | Y |
| `mse_trt` / `mae_trt` | TRT 路 vs GT 误差 | 仅 TRT 模式 |
| `mse_pt_trt` / `mae_pt_trt` | PT vs TRT 配对误差 | 仅 TRT 模式 |
| `mae_pt_trt_per_dim` / `mse_pt_trt_per_dim` | PT-TRT 配对逐维误差 | 仅 TRT 模式 |
| `mse_ptq` / `mae_ptq` | PTQ 路 vs GT 误差 | 仅 PTQ 模式 |
| `mse_pt_ptq` / `mae_pt_ptq` | PT vs PTQ 配对误差 | 仅 PTQ 模式 |
| `mae_pt_ptq_per_dim` / `mse_pt_ptq_per_dim` | PT-PTQ 配对逐维误差 | 仅 PTQ 模式 |

#### `timing` 字段内容

| 键 | 说明 |
|----|------|
| `infer_ms` | 总推理耗时 (ms) |
| `infer_ms_pt` | PyTorch 路耗时 (ms)（双路模式） |
| `infer_ms_trt` | TRT 路耗时 (ms)（TRT 对比模式） |
| `infer_ms_ptq` | PTQ 路耗时 (ms)（PTQ 对比模式） |

### InferResult

完整推理运行的结果集合。

```python
@dataclass
class InferResult:
    steps: list[StepResult]       # 所有 step 结果
    meta: dict[str, Any]          # 运行元信息
    start_index: int = 0          # 起始索引
    end_index_exclusive: int = 0  # 结束索引（不含）
```

### PredictionPack

推理后端返回的原始预测数据。

```python
@dataclass
class PredictionPack:
    pred_h: np.ndarray                        # 主路预测 (action_horizon, action_dim)
    gt_h: np.ndarray                          # 标注动作 (action_horizon, action_dim)
    pred_h_trt: np.ndarray | None = None      # TRT 路预测
    pred_h_ptq: np.ndarray | None = None      # PTQ 路预测
    infer_ms_pt: float = 0.0                  # 主路推理耗时 (ms)
    infer_ms_second: float | None = None      # 第二路推理耗时 (ms)
```

---

## 推理与后处理解耦

`infer_chunk` 与结果构建完全异步解耦：

```
推理线程                            后处理线程（可选）
─────────                          ────────────────
backend.predict()
  ↓ ChunkPayload（轻量 numpy 拷贝）
  ├─→ Queue ──→ ResultWorker       →  metrics 计算
  │              （仅 enable_result=True）  →  图像 JPEG 编码
  │                                →  StepResult 构建
  │                                →  on_step 回调
  └─→ 直接返回（不阻塞推理）
```

- **`enable_result=True`**：`run_all` 内部启动 `ResultWorker` 线程，从队列消费 `ChunkPayload`，异步完成 metrics 计算 + 图像编码 + `StepResult` 构建。推理线程只做 `predict()` + 轻量数据拷贝。
- **`enable_result=False`**：不启动后处理线程，不创建队列，推理路径零额外开销。适用于校准数据收集或纯吞吐测试。
- **WebSocket 模式**：`ws_server` 从 `infer_chunk` 拿到 `ChunkPayload` 后，在 ws 推送侧自行完成后处理（JSON 序列化 + 图像编码），通过 Janus 队列桥接 sync 推理线程与 async 事件循环。

---

## WebSocket 协议

### 离线推理 WebSocket

`WebSocketInferServer` 推送以下 JSON 消息类型：

### `meta` — 初始化元信息

```json
{
  "type": "meta",
  "run_id": "a1b2c3d4e5f6",
  "repo_id": "lerobot/libero_10",
  "backend": "pytorch",
  "mode": "pytorch",
  "action_horizon": 50,
  "start_index": 0,
  "end_index_exclusive": 500,
  "precision": "bf16",
  "rel_eps": 1e-8
}
```

双路模式额外包含 `pred1_name`、`pred2_name`、`pair_name` 以及 `tensorrt` / `ptq` 子对象。

### `step` — 推理步结果

```json
{
  "type": "step",
  "run_id": "a1b2c3d4e5f6",
  "episode_id": 0,
  "global_index": 0,
  "k_in_chunk": 0,
  "is_chunk_start": true,
  "action_horizon": 50,
  "prompt": "Pick up the red block",
  "gt_action": [0.1, 0.2, ...],
  "pred_action": [0.11, 0.19, ...],
  "metrics": {"mse": 0.001, "mae": 0.02, ...},
  "images": {"base_rgb_jpeg_b64": "/9j/4AAQ..."},
  "server_timing": {"infer_ms": 42.5}
}
```

`images` 和 `server_timing` 仅在 chunk 首 step (`k_in_chunk=0`) 中非 null。双路模式额外包含 `pred_action_trt` / `pred_action_ptq`。

### `gpu_stats` — GPU 监控

```json
{
  "type": "gpu_stats",
  "run_id": "a1b2c3d4e5f6",
  "device_index": 0,
  "gpu_util_pct": 85.0,
  "mem_util_pct": 62.0
}
```

### `server_progress` — 加载进度

```json
{
  "type": "server_progress",
  "run_id": "a1b2c3d4e5f6",
  "stage": "policy_pt",
  "message": "加载 PyTorch 策略（checkpoint → 内存/显存）…"
}
```

### `done` — 推理完毕

```json
{
  "type": "done",
  "phase": "finished",
  "run_id": "a1b2c3d4e5f6",
  "message": "推理完毕；server 即将关闭。",
  "start_index": 0,
  "end_index_exclusive": 500
}
```

### `error` — 异常

```json
{
  "type": "error",
  "run_id": "a1b2c3d4e5f6",
  "message": "..."
}
```

### 客户端控制命令

客户端可发送 JSON 控制推理暂停/恢复：

```json
{"type": "control", "action": "pause"}
{"type": "control", "action": "resume"}
```

服务端回复 `control_ack`：

```json
{"type": "control_ack", "action": "pause", "paused": true}
```

### PolicyServer WebSocket（openpi_client 协议）

`PolicyServer` 使用 **msgpack 二进制协议**（兼容 openpi_client），与上面的 JSON 协议完全不同。

#### 连接握手

```
客户端连接
  ← 服务端发送 metadata (msgpack)
    {
      "supports_score_endpoint": true/false,
      ... (策略元信息，如 action dims 等)
    }
```

#### 请求/响应循环

```
→ 客户端发送 obs (msgpack)
  {
    "image": np.ndarray,          # 观测图像
    "state": np.ndarray,          # 机器人状态
    "prompt": "task description", # 任务指令
    "_request_type": "infer",     # "infer" 或 "score"（可选，默认 "infer"）
    ...
  }

← 服务端返回 result (msgpack)
  {
    "actions": np.ndarray,        # 预测动作序列 (horizon, action_dim)
    "server_timing": {
      "process_ms": 42.5,         # 本次推理耗时 (ms)
      "prev_total_ms": 50.1       # 上一次请求总时间 (ms)
    },
    ...
  }
```

#### Score 请求

当 `serve.enable_score=true` 且后端支持 score 时，客户端可发送 `_request_type: "score"` 请求：

```
→ {"_request_type": "score", "image": ..., "state": ..., ...}
← {"value": float, "server_timing": {...}}
```

#### 健康检查

```
GET /healthz → 200 OK
```

#### 错误处理

推理异常时，服务端发送 traceback 文本并关闭连接（code=1011 INTERNAL_ERROR）。

---

## 校准数据收集

推理过程中可同时收集校准数据，用于后续 PTQ 量化。

在配置中设置 `calib` 节：

```json
{
  "calib": {
    "save_path": "/data/calib_output",
    "max_samples": 200,
    "item": "all"
  }
}
```

- `item` 支持：`"all"`（ViT + LLM + Expert + Denoise 全部收集）、`"vit"`、`"llm"`、`"expert"`、`"denoise"`
- 收集器在 `load()` 阶段创建，以 hook 方式挂载到模型子模块
- 推理运行中自动采集，`close()` 时停止并保存

---

## 使用示例

### 示例 1：PyTorch 浮点推理 + 结果回调

```python
from model_optimizer.infer.server import InferServer, load_config

config = load_config("config.json")
server = InferServer(config)
server.load()

def on_step(step):
    print(f"[ep={step.episode_id}] step {step.global_index}: "
          f"MSE={step.metrics['mse']:.6f}, MAE={step.metrics['mae']:.6f}")

result = server.run_all(on_step=on_step)
print(f"Total steps: {len(result.steps)}")
server.close()
```

config.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "config_name": "pi05_libero",
  "mode": "pytorch",
  "device": "cuda",
  "precision": "bf16",
  "dataset": {
    "num_samples": 100,
    "start_index": 0
  }
}
```

### 示例 2：PyTorch vs TensorRT 双路对比

```python
from model_optimizer.infer.server import InferServer, load_config

config = load_config("compare_config.json")
server = InferServer(config)
server.load()

result = server.run_all()
for step in result.steps:
    if step.is_chunk_start and step.timing:
        print(f"step {step.global_index}: "
              f"PT={step.timing['infer_ms_pt']:.1f}ms, "
              f"TRT={step.timing['infer_ms_trt']:.1f}ms, "
              f"PT-TRT MAE={step.metrics['mae_pt_trt']:.6f}")

server.close()
```

compare_config.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "mode": "pt_trt_compare",
  "device": "cuda",
  "precision": "bf16",
  "dataset": {"num_samples": 200},
  "tensorrt": {
    "engine_path": "/engines/pi05",
    "vit_engine": "vit.engine",
    "llm_engine": "llm.engine",
    "expert_engine": "expert.engine",
    "denoise_engine": "denoise.engine"
  }
}
```

### 示例 3：逐 chunk 手动推理（底层 API）

```python
from model_optimizer.infer.server import InferServer, load_config

config = load_config("config.json")
server = InferServer(config)
server.load()

# 手动遍历有效 chunk
for idx in range(server.start_index, server.end):
    payload = server.infer_chunk(idx)
    if payload is None:
        continue  # 不在对齐位置或跨 episode
    
    pack = payload.pack
    print(f"chunk idx={payload.idx}, episode={payload.episode_id}, "
          f"pred shape={pack.pred_h.shape}, infer={pack.infer_ms_pt:.1f}ms")

server.close()
```

### 示例 4：纯推理吞吐测试（禁用后处理）

```python
import time
from model_optimizer.infer.server import InferServer, load_config

config = load_config("config.json")
config.enable_result = False  # 不启动后处理线程

server = InferServer(config)
server.load()

t0 = time.perf_counter()
server.run_all()  # 返回 None，零后处理开销
elapsed = time.perf_counter() - t0

n = server.end - server.start_index
print(f"Inferred {n} frames in {elapsed:.2f}s ({n/elapsed:.1f} fps)")

server.close()
```

### 示例 5：WebSocket 流式推送

```python
from model_optimizer.infer.server import WebSocketInferServer, load_config

config = load_config("ws_config.json")
ws = WebSocketInferServer(config)
ws.run()  # 阻塞，推理完毕后自动关闭
```

ws_config.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "mode": "pytorch",
  "device": "cuda",
  "precision": "bf16",
  "dataset": {"num_samples": 500},
  "websocket": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 8765,
    "path": "/ws",
    "max_fps": 30,
    "history_size": 200,
    "gpu_stats_interval": 2.0,
    "jpeg_quality": 80,
    "send_wrist": true
  }
}
```

客户端连接 `ws://host:8765/ws` 后：
1. 收到 `meta` 或 `server_progress`（加载阶段）
2. 持续收到 `step` 消息流
3. 定期收到 `gpu_stats`
4. 推理结束收到 `done`，连接关闭

### 示例 6：PTQ 对比 + 校准数据收集

```python
from model_optimizer.infer.server import InferServer, load_config

config = load_config("ptq_calib_config.json")
server = InferServer(config)
server.load()  # 自动启动校准收集器

result = server.run_all()
for step in result.steps:
    if "mse_ptq" in step.metrics:
        print(f"step {step.global_index}: "
              f"PT MSE={step.metrics['mse_pt']:.6f}, "
              f"PTQ MSE={step.metrics['mse_ptq']:.6f}")

server.close()  # 自动保存校准数据
```

ptq_calib_config.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "mode": "pt_ptq_compare",
  "device": "cuda",
  "precision": "bf16",
  "dataset": {"num_samples": 300},
  "ptq": {
    "quant_cfg": "/configs/quant_int8.json",
    "calib_dir": "/data/calib",
    "parts": ["vit", "llm", "expert", "denoise"]
  },
  "calib": {
    "save_path": "/data/new_calib",
    "max_samples": 200,
    "item": "all"
  }
}
```

### 示例 7：异步事件循环中使用 WebSocket 服务

```python
import asyncio
from model_optimizer.infer.server import WebSocketInferServer, load_config

async def main():
    config = load_config("ws_config.json")
    ws = WebSocketInferServer(config)
    await ws.start()  # 异步入口，可与其他 coroutine 组合

asyncio.run(main())
```

### 示例 8：PolicyServer 本地策略服务

```python
from model_optimizer.infer.server import PolicyServer, load_config

config = load_config("serve_local.json")
server = PolicyServer(config)
server.load(on_progress=lambda s, m: print(f"[{s}] {m}"))
server.run()  # 阻塞，等待客户端连接
```

serve_local.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "config_name": "cfg_tianji_hard_0408_2.py",
  "mode": "pytorch",
  "device": "cuda",
  "precision": "bf16",
  "serve": {
    "host": "0.0.0.0",
    "port": 8000,
    "backend": "local",
    "robot_type": "unified_robot",
    "unify_action_mode": true,
    "default_prompt": "Fasten seatbelt"
  }
}
```

客户端连接：

```python
from openpi_client import WebsocketClientPolicy
policy = WebsocketClientPolicy(host="gpu-server", port=8000)
result = policy.infer({"image": img, "state": state, "prompt": "Fasten seatbelt"})
actions = result["actions"]  # (horizon, 16)
```

### 示例 9：PolicyServer 远程代理模式

当远程已有 openpi `serve_policy` 服务运行时，PolicyServer 可作为代理层：

```python
from model_optimizer.infer.server import PolicyServer, load_config

config = load_config("serve_remote.json")
server = PolicyServer(config)
server.load()
server.run()
```

serve_remote.json:

```json
{
  "checkpoint": "",
  "config_name": "",
  "mode": "pytorch",
  "serve": {
    "host": "0.0.0.0",
    "port": 9000,
    "backend": "remote",
    "remote_host": "gpu-server",
    "remote_port": 8000
  }
}
```

客户端连接本地 9000 端口，请求被透明转发到 `gpu-server:8000`。

### 示例 10：PolicyServer + TensorRT 加速

```python
from model_optimizer.infer.server import PolicyServer, load_config

config = load_config("serve_trt.json")
server = PolicyServer(config)
server.load()
server.run()
```

serve_trt.json:

```json
{
  "checkpoint": "/models/pi05/checkpoint",
  "config_name": "cfg_tianji_hard_0408_2.py",
  "mode": "tensorrt",
  "device": "cuda",
  "precision": "bf16",
  "tensorrt": {
    "engine_path": "/engines/pi05",
    "vit_engine": "vit.engine",
    "llm_engine": "llm.engine",
    "expert_engine": "expert.engine",
    "denoise_engine": "denoise.engine"
  },
  "serve": {
    "host": "0.0.0.0",
    "port": 8000,
    "backend": "local",
    "default_prompt": "Pick up the red block"
  }
}
```

### 示例 11：PolicyServer 异步入口

```python
import asyncio
from model_optimizer.infer.server import PolicyServer, load_config

async def main():
    config = load_config("serve_local.json")
    server = PolicyServer(config)
    server.load()
    await server.start()  # 异步入口，可与其他 coroutine 组合

asyncio.run(main())
```
