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

- **检查点**：目录内需有 `model.safetensors`，且包含 `assets/<asset_id>/` 下的 **归一化统计**（与 `create_trained_policy` 一致）。

## 基本用法（Libero 配置）

默认 `--config-name pi05_libero` 对应 `physical-intelligence/libero` 与 `LeRobotLiberoDataConfig`。

```bash
export PYTHONPATH="third_party/openpi/src:${PYTHONPATH}"

python scripts/deployment/pi05/lerobot_eval_compare.py \
  --checkpoint /path/to/checkpoint \
  --config-name pi05_libero \
  --num-samples 100 \
  --start-index 0 \
  --dataset-root ~/.cache/huggingface/lerobot/physical-intelligence/libero
```

常用参数说明：

| 参数 | 含义 |
|------|------|
| `--checkpoint` | 含 `model.safetensors` 与 `assets/` 的检查点路径 |
| `--config-name` | `openpi.training.config` 里已注册的 `TrainConfig` 名称 |
| `--num-samples` | 从 `--start-index` 起连续评估的帧数 |
| `--dataset-root` | 本地 LeRobot 根目录；省略则用 Hub 默认缓存（视 lerobot 版本而定） |
| `--pytorch-device` | 如 `cuda:0`；省略则自动选择 |
| `--seed` | NumPy 随机种子 |

输出日志中包含：样本数、平均 MSE、平均 MAE（及标准差）、**逐动作维度的 MAE**。

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
  --config-name your_custom_config_name \
  --num-samples 50
```

若 repack 后缺少 `actions` 或键名对不上，脚本会报错；此时应回到 **repack 映射** 与 **`action_sequence_keys`** 逐项核对数据集 `meta`/parquet 中的字段名。

## 相关文件

| 路径 | 说明 |
|------|------|
| `scripts/deployment/pi05/lerobot_eval_compare.py` | 评估入口 |
| `scripts/deployment/pi05/standalone_inference_script.py` | 单步推理 / 性能与保存 |
| `third_party/openpi/src/openpi/training/config.py` | `TrainConfig`、`LeRobotLiberoDataConfig` |
| `third_party/openpi/src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs`、`make_libero_example` |
| `third_party/openpi/src/openpi/policies/policy_config.py` | `create_trained_policy` |
| `third_party/openpi/src/openpi/training/data_loader.py` | `create_torch_dataset`、`transform_dataset` |
