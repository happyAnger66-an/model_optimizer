# Pi0.5 训练配置说明（OpenPI / Gr00t LeRobot）

本文说明在配套 **OpenPI / openpi_modified** 训练工程里，与 **Pi0.5 + LeRobot / Anyverse** 相关的若干配置字段（实现见 `training/base_cfg.py`、`training/data_loader.py`、`policies/gr00t_policy.py` 等）。

---

## 一、`assets` / `asset_id`

```python
assets=AssetsConfig(asset_id=ASSET_ID),
```

### `assets=AssetsConfig(asset_id=...)` 在做什么

`AssetsConfig` 用来指定**数据管线里「资产」从哪加载**——主要是预计算的 **`norm_stats`（归一化统计量）**；若启用相关 value-net 配置，还可能参与 **RL norm stats 钉扎** 等逻辑（见训练代码中的 `_inject_pinned_rl_norm_stats` / `_load_rl_norm_stats`）。

### `asset_id` 的含义

在 `DataConfigFactory.create_base_config` 中，有效 `asset_id` 为：

```text
asset_id = self.assets.asset_id or repo_id
```

- **显式设置了 `asset_id`**（例如 `ASSET_ID = "20260408"`）：到 **`assets_dirs / asset_id`**（或下面说的自定义 `assets_dir`）下加载 `norm_stats`（通常经下载/缓存辅助函数解析路径）。
- **未设置 `asset_id`**：回退为 **`repo_id`**（单数据集时为该字符串；`repo_id` 为列表时，加载逻辑会按实现拼接或选择子路径，见 `_load_norm_stats`）。

因此，在 **`repo_id` 为长列表、多数据源混合** 的训练配置里，把 `asset_id` 固定为某个字符串（如 `"20260408"`）的常见意图是：**所有 repo 共用同一份已生成好的 norm stats**，其文件放在名为该 `asset_id` 的资产子目录下，而不是让每个 repo 各自对应 `assets/<单个 repo_id>/`。

### `assets_dir`（可选）

`AssetsConfig` 还可设置 **`assets_dir`**：从**指定根目录**（例如某 base checkpoint 下的 `.../assets` 或集中存放资产的 URI）拉取资产；不设置时则使用训练入口传入的 **`assets_dirs`**（一般为当前实验 / checkpoint 下的 assets 根路径）。

### 与 checkpoint 的关系

设计上，这些资产在保存 checkpoint 时往往会**复制到 `assets/<asset_id>/`（或等价布局）**，便于复现以及**同一份统计量随权重一起分发**。

### 实操注意

- 需保证 **`assets_dirs / asset_id`**（或 `assets_dir / asset_id`）下已存在与**当前数据与动作空间**一致的 `norm_stats`；若缺失，训练侧通常会打日志并可能跳过加载（具体行为以后续数据管线为准）。
- `asset_id` 与 `repo_id` 解耦后，**统计量必须与混合数据分布匹配**；仅改名字而不重新计算统计量会导致归一化与真实分布不一致。

---

## 二、`Gr00tLerobotDataConfig`：`root_dir`、`repo_id`、`base_config` 与其它字段

对应典型写法（与业务配置中的片段一致）：

```python
Gr00tLerobotDataConfig(
    assets=AssetsConfig(asset_id=ASSET_ID),
    root_dir=ROOT_DIR,
    repo_id=REPO_ID,
    base_config=DataConfig(
        prompt_from_episode=True,
        action_sequence_keys=ACTION_SEQUENCE_KEYS,
        frame_attributes_preprocessors=FRAME_ATTRS_PREPROCESSORS,
    ),
    extra_delta_transform=False,
    use_delta_joint_actions=False,
    delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
    public_dataset_camera_map=_utils.PUBLIC_DATASET_MAP,
    align_dim=ALIGN_DIM,
    target_action_dim=TARGET_ACTION_DIM,
    unify_action_space=UNIFY_ACTION_SPACE,
    robot_type=ROBOT_TYPE,
    robot_align_info=RobotAlignInfo(robot_align_info=ROBOT_ALIGN_INFO),
)
```

### `root_dir`

LeRobot / 混合数据管线使用的**数据根目录**。`DataConfigFactory.create_base_config` 会把它写入最终 `DataConfig.root_dir`；具体与 `repo_id` 如何拼路径、是否走 Anyverse 多 repo，由 `data_loader` 中对应 `create_*_dataset` 分支决定（单 repo 时常为 `os.path.join(root_dir, repo_id)` 一类形式）。

### `repo_id`

- **字符串**：单个数据集标识（LeRobot `repo_id`）。
- **列表**：多数据源混合训练时，每个元素通常对应 `root_dir` 下的一个子路径 / 数据集名；采样与 norm 等逻辑需与训练入口一致。

### `base_config=DataConfig(...)`

`Gr00tLerobotDataConfig.create()` 里通过 `create_base_config(...)` 与 `dataclasses.replace(self.base_config or DataConfig(), ...)` 合并，把你在 `base_config` 里设的字段并入最终 `DataConfig`。常见子字段：

| 字段 | 作用 |
|------|------|
| **`prompt_from_episode`** | 为 `True` 时，在标准 LeRobot 数据加载路径上会包一层 `PromptFromEpisodeTask`，用 **episode 元数据里的 task 文本** 作为 prompt（与 `prompt_from_task` 二选一语义不同：后者来自 `dataset_meta.tasks` 的全局 task 表）。 |
| **`action_sequence_keys`** | 构造 `LeRobotDataset` 的 `delta_timestamps` 时使用的 **动作键名**元组（如 `("action",)`）。需与 HDF5/parquet 里实际字段名一致；长度与 `action_horizon` 共同决定滑窗时间戳。 |
| **`frame_attributes_preprocessors`** | 在 Anyverse 等数据集 **初始化阶段** 运行的处理器链：用于计算 **`valid_mask`、样本权重、分段/静态检测** 等帧级属性；为 `None` 或空时通常等价于「全帧有效、权重 1」。 |

### `extra_delta_transform`

为 `True` 时，在 `Gr00tLerobotDataConfig.create()` 里**额外**压入一组固定的 `DeltaActions` / `AbsoluteActions`（实现里使用 `make_bool_mask(6, -1)` 的模板），与下面 `use_delta_joint_actions` 是**两套开关**，一般不要与自定义 `delta_action_mask_indices` 混用除非你很清楚顺序与语义。

### `use_delta_joint_actions`

为 `True` 时，在数据变换管线中加入 **相对首帧状态的 delta 动作**（训练前向）及推理端的 **还原为绝对动作**：

- 若同时 **`use_semantic_delta_actions` 且 `unify_action_space` 且配置了 `robot_align_info`**，则走 **语义 delta**（按各 `robot_type` 的 state/action 维对齐关系构造 mask）。
- 否则使用 **`delta_action_mask_indices`** 调用 `make_bool_mask(*indices)` 生成布尔 mask（例如双机械臂常用多段 `[a,-1,b,-1]` 表示左臂若干维、`-1` 处为绝对量如夹爪等）。

为 `False` 时不做该组 delta 变换；若数据已是 delta 或希望全程绝对量，应保持 `False`。

### `delta_action_mask_indices`

仅在 **`use_delta_joint_actions=True`** 且未走语义 delta 分支时生效；传给 `make_bool_mask`，约定哪些动作维做 delta、哪些保持绝对（常见模式里 `-1` 表示该段最后一维为绝对）。

### `public_dataset_camera_map`

字典，交给 **`PublicDatasetMapTransform`**：把数据集中「公开命名」的相机键（如某第三方布局下的 `observation/front_image`）映射到策略 / 训练管线期望的键（如 `observation.images.head`）。需与 `repack_dict` 及实际 HDF 列名一致，否则图像进不了模型。

### `align_dim`

在 **语义 delta** 路径中与 `robot_align_info` 一起使用：对每个 `robot_type` 计算 state 与 action 维索引交集，并只保留 **小于 `align_dim`** 的维，生成 `SemanticDeltaActions` 的 mask。非语义 delta 时也会写入 `DataConfig`，供其它 transform / 数据逻辑读取。

### `target_action_dim`

传给 **`Gr00tLerobotOutputs`**：在从模型输出回到环境动作时，对 `actions` 做 **`[..., target_action_dim]`** 切片，即**只保留指定维度的动作子空间**（例如只要前 16 维关节/末端相关分量）。需与模型 `action_dim`、环境接口一致。

### `unify_action_space`

写入 `DataConfig`；在 **`use_semantic_delta_actions` + `use_delta_joint_actions`** 时参与是否按统一动作空间与 `robot_align_info` 构造语义 mask。多机种、多对齐表训练时常开。

### `robot_type`

传入 **`Gr00tLerobotInputs`**：声明当前样本对应的机器人类型 id（如 `unified_robot`），需与 **数据集里写入的 `robot_type` 字段** 及推理端一致，用于选择正确的 repack/对齐与（若启用）语义 delta 分支。

### `robot_align_info`

`RobotAlignInfo` 包装一层 **`robot_align_info` 字典**：描述各 `robot_type` 下 state/action 维名称与索引映射。用于 **语义 delta**、**统一动作空间** 以及与 `align_dim` 相关的维对齐；不启用语义 delta 时仍可能供其它 transform 使用。

---

## 文档范围说明

上述行为以 **openpi_modified** 中当前实现为准；若你本地 fork 改过 `create()` / `data_loader`，以实际代码为准。
