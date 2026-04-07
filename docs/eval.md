# LeRobot 离线评估：预测动作与标签误差

本文说明 `scripts/deployment/pi05/lerobot_eval_compare.py` 在做什么、如何运行，以及如何用 **非 Libero** 的 LeRobot 数据评估。

## 脚本做什么

在 **开源 LeRobot 格式** 的数据上，用与训练一致的检查点加载 **π0.5（openpi）策略**，对每一帧：

1. 从 `LeRobotDataset` 取出样本（含与 `action_horizon` 对齐的 **action chunk**），必要时注入 **任务文本**（`PromptFromLeRobotTask`）。
2. 仅应用与训练相同的 **repack**（把数据集中的键映射成策略期望的键，例如 `observation/image` 等）。
3. 去掉 `actions` 后作为 **`obs`** 调用 `policy.infer(obs)`。
4. 将 **`infer` 输出的 `actions`** 与 **repack 后数据里的 `actions`（标签）** 对齐维度后计算 **MSE / MAE**（含逐维 MAE）。

与 `openpi.training.data_loader.create_torch_dataset` 的构造方式一致（`delta_timestamps`、`prompt_from_task`），保证 **标签 chunk** 与训练时定义一致。

**注意**：`Policy.infer` 内部仍会走完整推理链（`LiberoInputs`、归一化、模型、`Unnormalize`、`LiberoOutputs` 等）。标签是 **数据集中未经过模型** 的 action；该对比衡量的是「当前观测下模型输出」与「数据里记录的将来动作」之间的数值差，**不是**在线闭环成功率。

## 依赖与环境

- **Python 包**：`lerobot`（与 openpi 所用 API 兼容）、`tyro`，以及 openpi 的其余依赖（见 `third_party/openpi`）。
- **源码路径**：需让 Python 能找到 `openpi`，例如：

```bash
export PYTHONPATH="/path/to/model_optimizer/third_party/openpi/src:${PYTHONPATH}"
```

- **model_optimizer（仅 TensorRT）**：`lerobot_eval_compare.py` 在 `--inference-mode tensorrt` 时会 import `model_optimizer.infer.tensorrt.pi05_executor`。请使用可编辑安装 `pip install -e .`，或把仓库的 `src` 加入 `PYTHONPATH`：

```bash
export PYTHONPATH="/path/to/model_optimizer/src:${PYTHONPATH}"
```

- **检查点**：目录内需有 `model.safetensors`，且包含 `assets/<asset_id>/` 下的 **归一化统计**（与 `create_trained_policy` 一致）。

## 基本用法（Libero 配置）

默认 `--config pi05_libero` 对应 `physical-intelligence/libero` 与 `LeRobotLiberoDataConfig`。

```bash
export PYTHONPATH="third_party/openpi/src:${PYTHONPATH}"

python scripts/deployment/pi05/lerobot_eval_compare.py \
  --checkpoint /path/to/checkpoint \
  --config pi05_libero \
  --num-samples 100 \
  --start-index 0 \
  --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero
```

常用参数说明：

| 参数 | 含义 |
|------|------|
| `--checkpoint` | 含 `model.safetensors` 与 `assets/` 的检查点路径 |
| `--config` | `openpi.training.config` 里已注册的 `TrainConfig` 名称 |
| `--num-samples` | 从 `--start-index` 起连续评估的帧数 |
| `--dataset-root` | 本地 LeRobot 根目录；省略则用 Hub 默认缓存（视 lerobot 版本而定） |
| `--device` | 如 `cuda:0`；省略则自动选择 |
| `--inference-mode` | `pytorch`（默认）或 `tensorrt` |
| `--engine-path` / `--vit-engine` / … | TensorRT 目录与各子引擎文件名；见下文「TensorRT 推理」 |
| `--precision` | `fp16` / `bf16` / `fp32`，仅 TensorRT 路径使用 |
| `--seed` | NumPy 随机种子 |

输出日志中包含：样本数、平均 MSE、平均 MAE（及标准差）、**逐动作维度的 MAE**。

### 按 episode 保存轨迹对比图（可选）

依赖底层 `LeRobotDataset.hf_dataset` 中的 `episode_index` 列（与全局帧下标对齐）。脚本在 `[start_index, start_index+num_samples)` 内仍对**每一帧**计算误差；若指定 `--trajectory-plot-dir`，会按 `episode_index` 分组，**每个 episode 输出一张 PNG**（每个动作维度一条子图：gt / pred，可选 `observation/state` 与动作维一致时叠加；红点间隔为 `action_horizon`）。

| 参数 | 含义 |
|------|------|
| `--trajectory-plot-dir` | 保存目录 |
| `--max-trajectory-plots` | 最多保存几张图；`0` 表示不限制 |
| `--plot-episodes` | 只画列出的 episode 编号；留空则画区间内出现的全部 episode |

若无法读取 `episode_index`，会退化为整段评估区间视为 `episode_id=0` 的一张图。

## TensorRT 推理（与 `standalone_inference_script.py` 对齐）

### `standalone_inference_script.py` 里 TensorRT 在做什么

1. 先用 `policy_config.create_trained_policy` 加载 **完整 PyTorch** `Policy`（含 `PI0Pytorch` 的 `policy._model`）。
2. 若 `inference_mode == "tensorrt"`，构造 `Pi05TensorRTExecutor(policy, precision)`，并按 CLI 传入的目录与文件名调用 `load_model(addict.Dict({...}))`。
3. `Pi05TensorRTExecutor._setup_trt_engine` 在 **`policy._model`（即 openpi 的 PI0Pytorch）上原地替换** 若干 `forward` / 方法：例如用 TRT `Engine` 替换 `get_image_features`、`language_model.forward`、`gemma_expert.model.forward`、`denoise_step` 等（取决于你提供了哪些引擎文件名）。
4. 主循环仍对 **`policy` 调用 `policy.infer(obs)`**（未改为 `executor.infer`）。因为被替换的是同一个 `policy._model` 对象，**实际计算已在 TRT 路径上**。

精度由 `precision`（`fp16` / `bf16` / `fp32`）传给 `Pi05TensorRTExecutor`。

### 在 `lerobot_eval_compare.py` 中使用 TensorRT

增加与 standalone 相同的选项即可，例如：

```bash
export PYTHONPATH="/path/to/model_optimizer/third_party/openpi/src:/path/to/model_optimizer/src:${PYTHONPATH}"

python scripts/deployment/pi05/lerobot_eval_compare.py \
  --checkpoint /path/to/checkpoint \
  --config pi05_libero \
  --inference-mode tensorrt \
  --engine-path /path/to/trt_engine_directory \
  --vit-engine your_vit.trt \
  --llm-engine your_llm.trt \
  --expert-engine your_expert.trt \
  --denoise-engine denoise.engine \
  --precision bf16 \
  --num-samples 50
```

- `--engine-path`：引擎文件所在**目录**（与各 `--*-engine` 文件名拼接成完整路径）。
- 某个子引擎 CLI 留空时，**不替换**该子模块，仍走对应 PyTorch 实现（与 standalone 行为一致）。

## 如何使用非 Libero 数据

非 Libero 数据集列名、图像键、状态维度和 **动作键名** 往往不同。脚本 **不会** 自动猜测列名；它完全依赖所选 **`TrainConfig` 派生出的 `DataConfig`**（尤其是 `repack_transforms`、`action_sequence_keys`、`prompt_from_task`）。要做的事可以概括为下面几步。

### 1. 在 openpi 中新增或复用 `TrainConfig`

在 `third_party/openpi/src/openpi/training/config.py` 的 `_CONFIGS` 中增加一项（或复制 `pi05_libero` 改一版），至少包含：

- **`model`**：与你的检查点一致（如 `Pi0Config(pi05=True, action_horizon=..., ...)`）。
- **`data`**：使用 **`DataConfigFactory`**（可参考 `LeRobotLiberoDataConfig`）：
  - **`repo_id`**：你的 LeRobot 数据集 ID 或本地等价标识。
  - **`repack_transforms`**：用 `RepackTransform` 把 **你数据里 flatten 后的路径** 映射到策略管线使用的键。  
    Libero 示例将 `image`、`wrist_image`、`state`、`actions`、`prompt` 映射到 `observation/image` 等（见 `LeRobotLiberoDataConfig.create`）。你的数据需映射到 **`LiberoInputs` 所读取的键**（`observation/image`、`observation/wrist_image`、`observation/state`、`prompt`），或你改用自定义 `*Inputs` 类时映射到对应键。
  - **`action_sequence_keys`**：参与 `delta_timestamps` 的动作键，默认常为 `("actions",)`；若数据集用 `action` 等名字，需在这里与 repack 一致。
  - **`prompt_from_task`**：若数据用 `task_index` 且 meta 里有任务表，可设为 `True`；否则在 repack 里提供 `prompt`，或依赖 `InjectDefaultPrompt`（由 `create_trained_policy` 的 `default_prompt` 注入）。
  - **`assets` / `asset_id`**：归一化统计所在位置需与检查点内 **`assets/<asset_id>`** 一致；用错会导致归一化与训练不一致。

### 2. 动作维度与 `LiberoOutputs`

当前 Libero 策略在输出侧将动作 **截断为前 7 维**。若你的数据集动作维度不同，需要：

- 在 **`libero_policy.LiberoOutputs`**（或你自定的 Outputs）中修改截断长度，使其与数据集一致；并保证 **模型 `action_dim` / 训练** 与之一致。

脚本里预测与标签在 **最后一维** 上取 `min(pred_dim, gt_dim)` 对齐；维数不一致时仅比较重叠部分，但 **语义是否正确** 仍依赖上述配置。

### 3. 可选：自定义 `Inputs` / `Outputs`

若无法把数据整理成 Libero 键名，可在 `openpi/policies/` 下复制 `LiberoInputs` / `LiberoOutputs` 为新类，在 `DataConfig` 的 `data_transforms` 中引用，并在 **`policy_config.create_trained_policy`** 使用的 config 与训练时 **完全一致**。

### 4. 运行评估

配置注册并安装好依赖后：

```bash
python scripts/deployment/pi05/lerobot_eval_compare.py \
  --checkpoint /path/to/your_checkpoint \
  --config your_custom_config_name \
  --num-samples 50
```

若 repack 后缺少 `actions` 或键名对不上，脚本会报错；此时应回到 **repack 映射** 与 **`action_sequence_keys`** 逐项核对数据集 `meta`/parquet 中的字段名。

## 相关文件

| 路径 | 说明 |
|------|------|
| `scripts/deployment/pi05/lerobot_eval_compare.py` | 评估入口 |
| `scripts/deployment/pi05/lerobot_eval_webui_server.py` | WebUI server（WebSocket 流式推送对齐 step + RGB + prompt） |
| `scripts/deployment/pi05/standalone_inference_script.py` | 单步推理 / 性能与保存 |
| `third_party/openpi/src/openpi/training/config.py` | `TrainConfig`、`LeRobotLiberoDataConfig` |
| `third_party/openpi/src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs`、`make_libero_example` |
| `third_party/openpi/src/openpi/policies/policy_config.py` | `create_trained_policy` |
| `third_party/openpi/src/openpi/training/data_loader.py` | `create_torch_dataset`、`transform_dataset` |

## WebUI（client-server，实时展示对齐推理效果）

`lerobot_eval_compare.py` 生成的是离线 PNG；若你希望像在线推理一样 **实时看到**（prompt + RGB + gt/pred 曲线滚动），可以启动 WebSocket server 并用浏览器订阅。

### 1) 启动 server（WebSocket 流式推送）

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

**行为说明**：进程启动后会 **立即监听** WebSocket 端口；数据集与策略在 **单线程推理池**（`ThreadPoolExecutor(max_workers=1)`）中加载与执行 `infer`，避免阻塞 asyncio 事件循环。先连上的 client 会收到 `type=meta` 且 `phase=loading` 的提示，加载结束后再广播完整 meta 与 step 流（PyTorch/CUDA 上下文固定在该推理线程中，避免多线程争用 GPU）。

关键参数：

| 参数 | 含义 |
|------|------|
| `--send-wrist` | 额外发送 `observation/wrist_image`（默认只发 base） |
| `--jpeg-quality` | JPEG 质量（默认 85） |
| `--max-fps` | 限制推送速率（step/s）；`0` 表示不限制 |
| `--history-size` | 缓存最近 N 条消息，新 client 连接后先回放（`0` 不缓存） |
| `--client-ws-url` | 浏览器默认 WebSocket 地址，写入 `webui_client/server_hint.json`；例如远端填 `ws://192.168.1.10:8765/ws`。不设且 `--host 0.0.0.0` 时为 `ws://127.0.0.1:{port}{path}` |
| `--inference-mode tensorrt` + 引擎参数 | 与 `lerobot_eval_compare.py` 一致 |

对齐语义（重要）：

- server **每隔 `action_horizon` 帧**做一次推理，得到 chunk `pred[0..H-1]`。
- 将每个 `k` 的 `pred[k]` 与 label `gt[k]` 对齐，并映射到连续全局帧 `global_index = idx + k`。
- 跨 episode 边界的 chunk 会跳过，避免把两段 episode 拼接。
- 本 run 在 `[start_index, end)` 内全部推完后，server 会再推送一条 **`type=done`**（含 `message` / `phase=finished` 等）；WebUI 会显示“推理已结束”。随后 **整个 server 进程会关闭 WebSocket 监听并退出**（需再次评估时请重新启动脚本）。

### 2) 打开 client（浏览器订阅显示）

client 是纯静态页面，路径：

- `scripts/deployment/pi05/webui_client/index.html`

推荐在该目录启动一个静态文件服务器：

```bash
cd scripts/deployment/pi05/webui_client
python -m http.server 8000
```

然后浏览器打开 `http://127.0.0.1:8000/`。页面加载时会尝试读取同目录的 `server_hint.json`（由 server 启动时生成）并自动填入 `ws_url`；也可手动修改。

- 本机示例：`ws://127.0.0.1:8765/ws`
- 局域网另一台机器打开页面时，请在启动 server 时指定 `--client-ws-url ws://<server_ip>:8765/ws`，或手动在页面填写该地址。

**主题**：页面顶部「主题」可选浅色 / 深色；偏好保存在浏览器 `localStorage`。

页面默认只显示 base RGB；若你在 server 侧开启 `--send-wrist`，client 勾选“显示 wrist”即可显示第二张图。
