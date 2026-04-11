# Pi0.5 部署流程：策略服务 + 机器人控制

本文档梳理 Pi0.5 模型从训练产物到真机部署的完整链路，涵盖两个核心入口：

- **serve_policy.py** — 策略推理服务端，加载模型并通过 WebSocket 提供 `infer` / `score` 接口
- **record_unified.py** — 机器人控制客户端，支持纯录制、纯推理、推理+录制三种模式

## 目录

- [系统架构](#系统架构)
- [策略服务端 serve_policy.py](#策略服务端-serve_policypy)
  - [启动流程](#启动流程)
  - [配置文件结构](#配置文件结构)
  - [数据变换管道](#数据变换管道)
  - [Pi0.5 推理核心](#pi05-推理核心)
  - [WebSocket 协议](#websocket-协议)
- [机器人控制端 record_unified.py](#机器人控制端-record_unifiedpy)
  - [三种运行模式](#三种运行模式)
  - [核心组件](#核心组件)
  - [异步推理流程](#异步推理流程)
  - [人工接管机制](#人工接管机制)
  - [子任务管理](#子任务管理)
- [客户端-服务端交互](#客户端-服务端交互)
- [关键设计](#关键设计)
- [运行示例](#运行示例)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        机器人控制端 (record_unified.py)               │
│                                                                     │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────┐  ┌───────────┐  │
│  │  Robot    │  │ PolicyInference  │  │ Action    │  │ SubTask   │  │
│  │ (硬件)    │  │ Client           │  │ Buffer    │  │ Manager   │  │
│  └────┬─────┘  └───────┬──────────┘  └─────┬─────┘  └───────────┘  │
│       │                │                    │                       │
│       │   observation  │  WebSocket         │  fusion               │
│       └────────────────┼────────────────────┘                       │
└────────────────────────┼────────────────────────────────────────────┘
                         │ msgpack
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                    策略服务端 (serve_policy.py)                      │
│                                                                    │
│  ┌──────────────────────┐   ┌──────────────────────────────────┐   │
│  │ WebsocketPolicyServer│   │           Policy                 │   │
│  │                      │──→│  input_transforms                │   │
│  │  recv → infer → send │   │    ↓                             │   │
│  └──────────────────────┘   │  Pi0.5 Model                     │   │
│                             │  (ViT + LLM + Expert + Denoise)  │   │
│                             │    ↓                             │   │
│                             │  output_transforms               │   │
│                             └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 策略服务端 serve_policy.py

### 启动流程

以如下命令为例：

```bash
python serve_policy.py \
    --policy.config=cfg_tianji_hard_0408_2.py \
    --policy.dir=checkpoints/cfg_tianji_hard_0408_2/cfg_tianji_hard_0408_2_exp/30000
```

完整调用链：

```
main(args: Args)
  │
  ├─ 1. create_policy(args)
  │     │
  │     ├─ args.policy = Checkpoint(config=".py 路径", dir="checkpoint 目录")
  │     │
  │     ├─ get_config("cfg_tianji_hard_0408_2.py")
  │     │   └─ 检测到 .py 后缀 → Config.fromfile() 动态加载
  │     │   └─ 返回文件中定义的 TrainConfig 实例
  │     │
  │     └─ create_trained_policy(train_config, checkpoint_dir, ...)
  │         │
  │         ├─ 2. 检测模型格式
  │         │   └─ 检查 checkpoint_dir/model.safetensors 是否存在
  │         │      ├─ 存在 → PyTorch 路径
  │         │      └─ 不存在 → JAX 路径（加载 params/）
  │         │
  │         ├─ 3. 加载模型权重
  │         │   ├─ PyTorch: model.load_pytorch() + to_bfloat16
  │         │   └─ JAX:     model.load(restore_params(..., dtype=bf16))
  │         │
  │         ├─ 4. 构建数据变换管道
  │         │   └─ Gr00tLerobotDataConfig.create()
  │         │
  │         ├─ 5. 加载归一化统计量
  │         │   └─ load_norm_stats(checkpoint_dir/assets, asset_id)
  │         │
  │         └─ 6. 组装 Policy 对象
  │             └─ Policy(model, transforms=[...], output_transforms=[...])
  │
  ├─ 7. 创建 WebSocket 服务器
  │     └─ WebsocketPolicyServer(policy, host="0.0.0.0", port=8000)
  │
  └─ 8. server.serve_forever()  ← 阻塞
```

### 配置文件结构

以 `cfg_tianji_hard_0408_2.py` 为例，该配置定义了一个 **Pi0.5 双臂安全带操作任务**：

```python
cfg = TrainConfig(
    name     = "cfg_tianji_hard_0408_2",
    model    = Pi0Config(pi05=True, action_horizon=30, max_token_len=156),

    rtc_max_delay = 8,  # RTC 最大延迟步数

    data = Gr00tLerobotDataConfig(
        root_dir   = "/mnt/oss_data/anyverse_tianji/origin/",
        repo_id    = [27 个数据集],  # 多数据集混合训练
        # 统一动作空间：左臂 7 关节 + 1 gripper + 右臂 7 关节 + 1 gripper = 16 维
        unify_action_space       = True,
        robot_type               = "unified_robot",
        delta_action_mask_indices = [7, -1, 7, -1],
        align_dim                = 16,
        target_action_dim        = range(16),
        # 数据预处理
        frame_attributes_preprocessors = [
            VelocityBasedStaticDetector(...),       # 标记静态帧
            PruneHeadTailStaticValidMaskPreprocessor(...),  # 裁剪首尾静态帧
        ],
    ),

    weight_loader  = CheckpointWeightLoader("pi05_base/params"),
    batch_size     = 256,
    num_train_steps = 30000,
    peak_lr        = 2e-5,
)
```

#### 关键配置项

| 配置 | 值 | 说明 |
|------|-----|------|
| `pi05=True` | - | 启用 Pi0.5 架构（含 Expert + subtask） |
| `action_horizon=30` | 30 步 | 每次推理预测未来 30 步动作 |
| `rtc_max_delay=8` | 8 步 | RTC 模式允许客户端传入至多 8 步已执行动作 |
| `unify_action_space=True` | - | 多种机器人映射到统一 16 维动作空间 |
| `delta_action_mask_indices=[7,-1,7,-1]` | - | 7 关节用 delta 动作，1 gripper 用绝对动作，左右各一组 |
| `align_dim=16` | 16 维 | 统一动作空间总维度 |

### 数据变换管道

`create_trained_policy` 内部通过 `Gr00tLerobotDataConfig.create()` 构建输入/输出变换链。推理时数据经过如下流水线：

```
客户端 obs_dict
  │
  ▼ ─── Input Transforms ───
  │
  ├─ InjectDefaultPrompt       obs 中无 prompt 时注入默认值
  │
  ├─ Gr00tLerobotInputs        统一动作空间转换
  │   ├─ 图像: uint8 → float32 归一化
  │   ├─ state: 按 robot_type 对齐到 align_dim (16 维)
  │   └─ action: 对齐到统一动作空间
  │
  ├─ Normalize                  z-score 归一化（使用训练集统计量）
  │
  └─ ModelTransforms.inputs     tokenize prompt + 图像 resize/padding
  │
  ▼ ─── Pi0.5 Model ───
  │
  model.sample_actions()
  │
  ▼ ─── Output Transforms（逆序）───
  │
  ├─ ModelTransforms.outputs    模型输出后处理
  ├─ Unnormalize                反归一化 → 恢复物理量纲
  └─ Gr00tLerobotOutputs        从统一空间取 target_action_dim (前 16 维)
  │
  ▼
返回 {"actions": ndarray(30, 16), "policy_timing": {...}}
```

### Pi0.5 推理核心

`model.sample_actions()` 的内部流程：

```
observation (images + state + prompt)
  │
  ├─ [可选] sample_subtask()
  │   └─ 自回归解码 subtask tokens（低频运行），注入 observation
  │
  ├─ embed_prefix(observation) → prefix_tokens
  │   └─ ViT 编码图像 + tokenize prompt + state embedding
  │
  ├─ PaliGemma LLM forward → KV cache（缓存前缀表示）
  │
  └─ Flow Matching 去噪（10 步 Euler 积分）:
      │
      for t in [1.0 → 0.0], dt = -0.1:
        ├─ x_t = where(rtc_mask, action_prefix, x_t)    ← RTC: 前 delay 步锁定
        ├─ embed_suffix(x_t, time_cond) → suffix_tokens
        ├─ Expert denoise + LLM cross-attention (复用 KV cache)
        └─ x_{t+dt} = x_t + dt * predicted_velocity     ← Euler step
      │
      └─ x_0 → actions (30, 16)
```

**RTC (Real-Time Control)** 机制：客户端传入已执行的动作前缀 `action_prefix` 和延迟步数 `delay`，去噪时前 `delay` 步动作被锁定为已知值（`time_cond=0`，无噪声），模型仅预测剩余部分。

### WebSocket 协议

服务端使用 `msgpack_numpy` 序列化，处理循环：

```
客户端连接
  │
  ├─ 服务端发送 metadata (policy_metadata dict)
  │
  └─ 请求循环:
      ├─ 接收 msgpack → obs_dict
      │   ├─ _request_type = "infer" (默认) → policy.infer(obs)
      │   └─ _request_type = "score"        → policy.score_observation(obs)
      │
      ├─ 附加 server_timing:
      │   ├─ process_ms:    本次推理耗时
      │   └─ prev_total_ms: 上次请求总耗时（含网络）
      │
      └─ 发送 msgpack → result_dict
```

健康检查端点：`GET /healthz → 200 OK`

---

## 机器人控制端 record_unified.py

### 三种运行模式

| 模式 | 数据集 | 策略推理 | 遥操作 | 时间限制 | 场景 |
|------|:---:|:---:|:---:|:---:|------|
| `record` | Y | - | Y | `episode_time_s` | 人工遥操作采集训练数据 |
| `infer` | - | Y | - | 无限（手动停止） | 策略部署到真机验证 |
| `infer_record` | Y | Y | Y（接管时） | `episode_time_s` | 策略推理同时录制，可人工接管 |

主入口流程：

```
record(cfg: RecordConfig)
  │
  ├─ 初始化
  │   ├─ TTSService       语音播报（自动探测外置 TTS / 回退系统 say）
  │   ├─ AsyncLogger      后台线程异步写日志文件
  │   ├─ Robot             机器人连接
  │   ├─ Teleoperator      遥操作器（可选）
  │   ├─ PolicyInferenceClient  WebSocket 策略客户端（可选）
  │   ├─ LeRobotDataset    数据集（可选，纯推理模式不创建）
  │   └─ KeyboardListener  键盘事件监听
  │
  ├─ mode="infer"
  │   └─ run_inference_only_loop()      ← 无数据集，无限循环
  │
  └─ mode="record" / "infer_record"
      └─ run_recording_session()        ← 多 episode 管理
          └─ run_record_loop()          ← 单 episode 控制循环
```

### 核心组件

#### PolicyInferenceClient

通过 `openpi_client.WebsocketClientPolicy` 与服务端通信：

```python
# 构造观测数据
obs_dict = {
    "observation/front_image":    head 相机图像 (CHW),
    "observation/wrist_image":    右腕相机图像 (CHW),
    "observation/wrist_image_lf": 左腕相机图像 (CHW),
    "observation/state":          关节位置 (float32 数组),
    "prompt":                     自然语言任务描述,
    # 可选 RTC 字段
    "action":                     已执行动作前缀,
    "action_mask":                前缀掩码,
    "infer_delay":                延迟步数,
}

# 远程推理
result = self.policy.infer(obs_dict)
actions = result["actions"]   # shape: (action_horizon, action_dim)
```

#### ActionBuffer

异步推理的核心数据结构，实现时间对齐和动作融合：

```
推理线程返回 action_chunk (30, 16)
                    │
        update_from_inference()
           ├─ 计算延迟: latency_steps = current_step - start_step
           ├─ 截掉已过期动作: valid = chunk[latency_steps:]
           ├─ 重叠部分加权融合:
           │    buffer[i] = buffer[i] * w + valid[i] * (1 - w)
           └─ 超出部分直接追加
                    │
        get_next_action()
           └─ 弹出 buffer[0]，整体前移，available_cnt -= 1
```

融合权重策略：
- **linear** — 从 `start_weight=0.9` 线性衰减到 `end_weight=0.1`
- **exponential** — `w = exp(-(i+0.5) / decay)`，近处更信任旧值

#### InferenceState

封装单次推理会话的全部资源：

```
InferenceState
  ├─ ActionBuffer            动作缓冲区
  ├─ input_queue             推理请求队列（async: maxsize=1）
  ├─ output_queue            推理结果队列
  ├─ inference_thread        后台推理工作线程
  ├─ global_step_id          全局步数（单调递增，跨接管保持）
  ├─ is_human_intervention   人工接管标志
  └─ transition_weight       接管恢复过渡权重 (0.9 → 0)
```

初始化时执行 warmup 推理，填充 ActionBuffer 初始值。

### 异步推理流程

`run_inference_only_loop`（纯推理）和 `run_record_loop`（录制模式推理部分）共享相同的异步推理逻辑：

```
主控制循环 (30fps)                    推理工作线程
────────────────                     ─────────────
step 0:  投递请求 ──────────────→     开始推理 (step 0)
step 1:  buffer.get_next_action()
step 2:  buffer.get_next_action()
  ...
step 19: buffer.get_next_action()
step 20: 投递新请求 ────────────→     ←── 返回结果 (step 0)
         update_from_inference()      开始推理 (step 20)
step 21: buffer.get_next_action()
  ...
```

主循环伪代码：

```python
while not stop_recording:
    observation = robot.get_observation()

    # 每 infer_interval 步投递一次推理请求
    if step_id % infer_interval == 0:
        action_prefix, delay = buffer.get_future_actions(default_infer_delay)
        queue_put_replace_oldest(input_queue, (prompt, obs, prefix, delay, step_id))

    # 非阻塞检查推理结果
    if not output_queue.empty():
        new_chunk, start_step = output_queue.get_nowait()
        buffer.update_from_inference(new_chunk, start_step, step_id, enable_fusion=...)

    # 等待 buffer 非空（超时 3s 告警）
    while buffer.is_empty():
        time.sleep(0.001)

    action = buffer.get_next_action()
    robot.send_action(action)
    busy_wait(1/fps - elapsed)
```

`queue_put_replace_oldest`：当推理队列已满（推理跟不上控制频率），丢弃最旧的请求，避免阻塞控制循环。

### 人工接管机制

仅 `infer_record` 模式支持，通过键盘事件 `switch_infer_mode` 触发：

```
键盘触发 switch_infer_mode
  │
  ├─ 进入接管:
  │   1. 播报 "停止模型推理"
  │   2. 等待 waiting_intervention_time_s (2s)
  │   3. 姿态同步 pose_sync_loop:
  │      └─ 从臂当前位置线性插值到主臂姿态 (pose_sync_duration_s = 3s)
  │   4. 开启重力补偿模式 (set_gravity_compensation_mode)
  │   5. 播报 "开始接管" → 人工遥操作控制
  │
  └─ 退出接管:
      1. 锁定当前关节位置 (robot.send_action(current_pose))
      2. 播报 "请立即撤离" + 等待 waiting_evacuation_time_s (2s)
      3. 重新初始化 ActionBuffer（新 warmup 推理）
      4. 设置 transition_weight = 0.9
      5. 播报 "恢复模型推理"
      6. 后续 transition_steps (15 步) 内逐步过渡:
         action = (1 - w) * policy_action + w * current_joint_pos
         w 从 0.9 线性衰减到 0 → 平滑恢复策略控制
```

### 子任务管理

`SubTaskManager` 将单个 episode 按时间划分为多个子任务阶段：

```python
# 配置
sub_task_durations = [5, 20, 10, 15]

# 内部计算累积时间戳
timestamps = [5, 25, 35, 50]  # 总时长 50s

# 每帧调用 update(timestamp)
#   timestamp < 5   → 子任务 0，首次进入播报 "开始第 1 个步骤"
#   5 ≤ t < 25      → 子任务 1
#   25 ≤ t < 35     → 子任务 2
#   35 ≤ t < 50     → 子任务 3
#   t ≥ 50          → 播报 "结束所有步骤"
```

子任务索引写入数据集的 `sub_task_index` 字段，用于后续训练中区分不同阶段。

### 相机配置

命令行直接传设备 ID，内部自动构建 `OpenCVCameraConfig`：

```
--head_camera=10    → cameras["head"]        = OpenCVCameraConfig(index=10, w, h, fps)
--left_camera=4     → cameras["left_wrist"]  = OpenCVCameraConfig(index=4, w, h, fps)
--right_camera=16   → cameras["right_wrist"] = OpenCVCameraConfig(index=16, w, h, fps)
```

ID 为 -1 表示禁用。支持通过 `--camera_width`、`--camera_height`、`--camera_fps` 统一设置分辨率和帧率。

---

## 客户端-服务端交互

完整的单次推理数据流：

```
record_unified.py                             serve_policy.py
─────────────────                             ─────────────────

robot.get_observation()
  ├─ head 相机图像 (HWC uint8)
  ├─ left/right_wrist 相机图像
  └─ 关节位置 (float)
        │
PolicyInferenceClient.get_action()
  ├─ 图像 HWC → CHW (einops)
  ├─ 拼接 state 向量
  ├─ 构造 obs_dict:
  │   observation/front_image     (CHW)
  │   observation/wrist_image     (CHW)
  │   observation/wrist_image_lf  (CHW)
  │   observation/state           (float32[])
  │   prompt                      (str)
  │   action                      (prefix, 可选)
  │   action_mask                 (可选)
  │   infer_delay                 (int, 可选)
  │
  │ ──── WebSocket msgpack ──────────────→
  │                                        WebsocketPolicyServer._handler()
  │                                          unpackb_with_writable(data)
  │                                          │
  │                                        policy.infer(obs)
  │                                          ├─ InjectDefaultPrompt
  │                                          ├─ Gr00tLerobotInputs
  │                                          ├─ Normalize (z-score)
  │                                          ├─ ModelTransforms
  │                                          │
  │                                          ├─ model.sample_actions()
  │                                          │   ├─ [subtask inference]
  │                                          │   ├─ embed_prefix → LLM → KV cache
  │                                          │   └─ 10-step flow matching denoise
  │                                          │       └─ actions (1, 30, 16)
  │                                          │
  │                                          ├─ ModelTransforms.outputs
  │                                          ├─ Unnormalize
  │                                          └─ Gr00tLerobotOutputs
  │                                          │
  │                                        result = {
  │                                          "actions": ndarray(30, 16),
  │                                          "policy_timing": {"infer_ms": ...},
  │                                          "server_timing": {"process_ms": ...},
  │                                        }
  │ ←─── WebSocket msgpack ──────────────
  │
  result["actions"]  →  ndarray(30, 16)
        │
ActionBuffer.update_from_inference()
  ├─ 对齐延迟，截掉过期动作
  └─ 与已有 buffer 加权融合
        │
ActionBuffer.get_next_action()
  └─ 弹出 buffer[0] → {joint_key: float, ...}
        │
robot.send_action(action)
```

---

## 关键设计

### 1. RTC (Real-Time Control)

解决异步推理延迟导致动作不连续的问题：

```
时间线:  t0 ──── 推理延迟 ──── t1
         │                     │
客户端:  发送请求(obs@t0)       收到结果
         继续执行 buffer 动作    update_from_inference
         │                     │
服务端:               policy.infer()
                      ├─ action_prefix = 客户端传来的已执行动作
                      ├─ delay = t1 - t0 (步数)
                      └─ 去噪时前 delay 步锁定为已知值
                         仅预测 delay 之后的未来动作
```

`rtc_max_delay=8` 意味着最多容忍 8 步（约 0.27s @30fps）的推理延迟。

### 2. 统一动作空间

多种机器人通过 `robot_type` 映射到统一 16 维空间：

```
unified_robot:  [L_j1..L_j7, L_grip, R_j1..R_j7, R_grip]  = 16 维
                   delta(7)   abs(1)    delta(7)   abs(1)

delta_action_mask_indices = [7, -1, 7, -1]
  → 前 7 维 delta，第 8 维 abs，再 7 维 delta，第 16 维 abs
```

推理时通过 `Gr00tLerobotInputs` / `Gr00tLerobotOutputs` 自动处理对齐。

### 3. 服务端无状态

每次请求独立处理，无会话状态。所有状态管理在客户端完成：
- 动作缓冲和融合 → `ActionBuffer`
- 接管状态和过渡 → `InferenceState`
- 请求节奏控制 → `infer_interval` + `queue_put_replace_oldest`

### 4. 推理与执行并行

异步模式下推理线程独立于控制循环：
- 控制循环以 30fps 恒定频率运行，从 buffer 取动作
- 推理线程按 `infer_interval` 节奏异步执行
- `input_queue maxsize=1`：推理跟不上时丢弃旧请求而非阻塞
- `ActionBuffer` 融合确保新旧动作平滑过渡

### 5. 安全接管

三重保障防止接管切换时机器人突变：
1. **姿态同步** — 线性插值 3s 将从臂移到主臂位置
2. **撤离等待** — 恢复前等待 2s 确保人员撤离
3. **渐进过渡** — `transition_weight` 从 0.9 衰减到 0（15 步），混合当前关节与策略动作

---

## 运行示例

### 启动策略服务

```bash
# 从 checkpoint 加载
python serve_policy.py \
    --policy.config=cfg_tianji_hard_0408_2.py \
    --policy.dir=checkpoints/cfg_tianji_hard_0408_2/cfg_tianji_hard_0408_2_exp/30000 \
    --host=0.0.0.0 \
    --port=8000

# 使用默认策略
python serve_policy.py --env=lerobot
```

### 纯推理（无录制）

```bash
python record_unified.py \
    --robot.type=arxx5_bimanual \
    --mode=infer \
    --inference_mode=async \
    --head_camera=10 \
    --left_camera=4 \
    --right_camera=16 \
    --policy_host=localhost \
    --policy_port=8000 \
    --dataset.single_task="Fasten seatbelt"
```

### 纯遥操作录制

```bash
python record_unified.py \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --mode=record \
    --left_camera=4 \
    --head_camera=10 \
    --right_camera=16 \
    --dataset.root=/data/recordings \
    --dataset.repo_id=seatbelt_demo \
    --dataset.single_task="Fasten seatbelt" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=45
```

### 推理 + 录制（带子任务和人工接管）

```bash
python record_unified.py \
    --robot.type=arxx5_bimanual \
    --mode=infer_record \
    --inference_mode=async \
    --head_camera=10 \
    --left_camera=4 \
    --right_camera=16 \
    --policy_host=localhost \
    --policy_port=8000 \
    --sub_task_durations="[5, 20, 10, 15]" \
    --action_horizon=30 \
    --infer_interval=20 \
    --fusion_type=exponential \
    --fusion_exp_decay=2.0 \
    --dataset.root=/data/recordings \
    --dataset.repo_id=seatbelt_infer \
    --dataset.single_task="Fasten seatbelt" \
    --dataset.episode_time_s=50
```

运行过程中通过键盘切换人工接管（`switch_infer_mode` 快捷键），可随时介入并恢复策略控制。
