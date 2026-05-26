# GenieSim 运行控制链路分析

本文重点分析 GenieSim 中“模型输出 action 后，如何模拟实际机器人控制执行”的实现路径。这里不展开具体控制算法细节，只说明主要数据流、模块职责和与真实机器人控制的对应关系。

## 总体结论

GenieSim 的 benchmark 推理主链路可以概括为：

```text
模型服务
  -> PiPolicy
  -> PiEnv / AbsPoseEnv
  -> APICore
  -> UIBuilder
  -> Isaac Sim Articulation
  -> PhysX 仿真执行
```

也就是说，模型输出的 action 会先被解释成“关节目标”或“末端位姿目标”，然后转换成 Isaac Sim 中机器人 articulation 的关节控制命令，最终由 PhysX 仿真推进机器人运动。

需要注意的是，benchmark 主链路并不是完整复刻真实机器人的底层控制栈。它更像是将模型输出映射为仿真机器人可执行的目标状态，再通过 Isaac Sim 的 articulation/drive 机制模拟执行结果。

## 主运行循环

主入口在：

- `/home/zhangxa/codes/genie_sim/source/geniesim/app/app.py`

该文件负责：

- 创建 Isaac Sim `World`
- 启用 ROS2 bridge
- 创建 `UIBuilder`
- 创建 `APICore`
- 启动 `TaskManager`
- 注册 physics callback
- 在主循环中持续调用 `world.step(render=True)`

关键逻辑是 physics callback 每个物理步会调用：

```text
api_core.physics_step()
api_core.on_ros_tick(step_size)
```

其中：

- `physics_step()` 负责消费待执行的物理控制命令。
- `on_ros_tick()` 负责 ROS 节点 spin、clock、joint state、TF、相机等状态发布。

这意味着模型推理线程不会直接修改仿真状态，而是通过 `APICore` 将控制请求排到 physics loop 中执行。

## Benchmark 推理链路

benchmark 的执行主逻辑在：

- `/home/zhangxa/codes/genie_sim/source/geniesim/benchmark/task_benchmark.py`

核心流程是：

```text
env.reset()
while running:
    observation = 当前观测
    action = policy.act(observation)
    observation, done, need_update, progress = env.step(action)
```

其中：

- `policy.act()` 调用模型或策略，得到 action。
- `env.step(action)` 将 action 转成机器人控制命令。
- `DataCourier` 负责从 `APICore` 取相机、深度、关节状态等观测。

## 模型 action 如何产生

模型策略类在：

- `/home/zhangxa/codes/genie_sim/source/geniesim/benchmark/policy/pipolicy.py`

`PiPolicy` 会把当前 observation 组织成 websocket payload，发送给模型服务。payload 主要包含：

- `state`：机器人关节状态，补齐/截断到 32 维。
- `eef`：左右末端位姿。
- `images`：头部、左手、右手相机图像。
- `depth`：对应深度图。
- `prompt`：任务指令。
- `task_name` / `episode_idx`：任务上下文。

模型服务返回：

```text
result["actions"]
```

这些 action 会被放入 `action_buffer`。之后每个 step 从 buffer 中取一个 action 执行；当 buffer 为空时，再次请求模型服务。

## action 的主要格式

benchmark 中主要支持两种 action 解释方式。

### 1. `pi` / `abs_joint`

实现位置：

- `/home/zhangxa/codes/genie_sim/source/geniesim/benchmark/envs/pi_env.py`

这类 action 直接表示关节目标：

```text
action[0:14]   -> 左右臂 14 个关节目标，通常左右臂各 7 个
action[14:16]  -> 左右夹爪动作
raw_action[20] -> G2 机器人可选腰部控制
```

执行时会调用：

```text
process_action(..., type="abs_joint")
relabel_gripper_action(...)
api_core.set_joint_positions(...)
```

当前 `abs_joint` 的后处理基本是直接返回 action 本身；代码中有平滑逻辑，但被提前 `return action` 短路。

### 2. `abs_pose`

实现位置：

- `/home/zhangxa/codes/genie_sim/source/geniesim/benchmark/envs/abs_pose_env.py`
- `/home/zhangxa/codes/genie_sim/source/geniesim/utils/ikfk_utils.py`

这类 action 表示左右末端执行器目标位姿：

```text
左手 EE: xyz + rpy
右手 EE: xyz + rpy
左右夹爪: 2 维
可选腰部控制
```

执行时会先通过 `IKFKSolver` 做 IK，将末端位姿转换为左右臂 14 个关节目标，然后再走关节控制链路。

## action 到仿真执行

无论是 `abs_joint` 还是 `abs_pose`，最终都会调用：

```text
APICore.set_joint_positions(...)
```

实现位置：

- `/home/zhangxa/codes/genie_sim/source/geniesim/app/controllers/api_core.py`

关键点是 `set_joint_positions()` 并不直接修改 articulation，而是调用：

```text
run_on_physics_loop(self._set_joint_positions, ...)
```

这会把控制请求放入 physics queue。主仿真 loop 的 physics callback 中调用 `api_core.physics_step()`，再真正执行 `_set_joint_positions()`。

最终调用路径是：

```text
APICore.set_joint_positions()
  -> APICore.run_on_physics_loop()
  -> APICore._set_joint_positions()
  -> APICore._joint_moveto()
  -> UIBuilder._move_to()
```

最终落点在：

- `/home/zhangxa/codes/genie_sim/source/geniesim/app/workflow/ui_builder.py`

`UIBuilder._move_to()` 会根据 `is_trajectory` 选择执行方式：

```text
is_trajectory=False:
    articulation.set_joint_positions(...)

is_trajectory=True:
    action = ArticulationAction(joint_positions=...)
    articulation.apply_action(action)
```

benchmark 推理中通常使用 `is_trajectory=True`，因此模型 action 最终会变成 Isaac Sim articulation 的 joint position action，由 PhysX/Articulation drive 模拟机器人运动。

## 观测反馈链路

模型下一步 action 依赖当前仿真状态。观测主要通过：

- `/home/zhangxa/codes/genie_sim/source/geniesim/utils/data_courier.py`
- `/home/zhangxa/codes/genie_sim/source/geniesim/app/ros_publisher/robot_interface.py`

获取内容包括：

- 关节状态
- 相机 RGB
- 深度图
- 左右末端位姿
- ROS joint state / TF / clock

`PiEnv.get_observation()` 会从 `DataCourier` 获取这些信息，并组织成模型输入。

## Data Collection 控制链路

除了 benchmark 推理链路，`source/data_collection` 里还有一套更接近“机器人运动控制器”的执行路径：

```text
任务 JSON / ActionScript
  -> DataCollectionAgent.step()
  -> target_gripper_pose / motion_type / gripper_action
  -> IsaacSimRpcRobot.move_pose()
  -> gRPC
  -> CommandController
  -> cuRobo / IK / Ruckig / articulation
```

关键文件：

- `/home/zhangxa/codes/genie_sim/source/data_collection/client/agent/omniagent.py`
- `/home/zhangxa/codes/genie_sim/source/data_collection/client/robot/omni_robot.py`
- `/home/zhangxa/codes/genie_sim/source/data_collection/server/grpc_server.py`
- `/home/zhangxa/codes/genie_sim/source/data_collection/server/command_controller.py`
- `/home/zhangxa/codes/genie_sim/source/data_collection/server/ui_builder.py`

这条链路中，高层动作会先解析成抓手目标位姿、运动类型、夹爪开合等，再通过 gRPC 发送给仿真 server。server 侧 `CommandController` 会根据命令类型执行：

- `LINEAR_MOVE`：末端目标位姿控制。
- `SET_JOINT_POSITION`：关节位置控制。
- `SET_GRIPPER_STATE`：夹爪 open/close。
- `GET_IK_STATUS`：IK 可达性查询。

末端运动可以走 cuRobo 避障/规划，也可以走轨迹插值/Ruckig 风格控制，最后仍然落到 Isaac articulation 上。

这条链路比 benchmark 推理链路更像真实机器人应用中的“高层任务规划 -> 运动规划 -> 控制执行”结构。

## Teleop 控制链路

遥操作链路在：

- `/home/zhangxa/codes/genie_sim/source/teleop/teleop.py`

典型流程是：

```text
Pico / VR 输入
  -> TeleOp.parse_arm_control()
  -> ROS retarget / joint command
  -> 仿真或外部控制端消费
```

它会解析：

- 左右手 delta pose
- 夹爪开合
- 腰部控制
- 底盘轮控制
- reset / playback / recording 信号

这部分更接近真实机器人遥操作接口，因为它通过 ROS topic 发布控制意图。但在当前 Python 代码中，完整消费者可能位于外部 ROS controller、Isaac 插件或二进制组件中。

## 与真实机器人控制的对应关系

相似点：

- action 抽象接近真实机器人控制接口：关节、末端、夹爪、腰部、底盘。
- observation 也接近真实机器人：相机、深度、关节状态、TF、末端位姿。
- 使用 ROS2 发布 `/joint_states`、TF、clock、camera 等信息。
- data collection 链路中包含 IK、运动规划、轨迹执行、夹爪控制。

差异点：

- benchmark 主链路没有完整底层伺服、电机、力矩、电流或真实 WBC 闭环。
- 模型 action 最终主要变成 Isaac articulation 的 position target。
- 夹爪在 benchmark 中主要是关节位置映射，data collection 中才有更明确的 open/close 模拟。
- 仿真中还存在 attach/detach object、playback、reset physics 等真实机器人没有的辅助机制。

## 简化理解

可以把 GenieSim 的控制执行理解成三层：

```text
模型层:
    根据图像、状态、指令输出 action

控制解释层:
    将 action 解释为关节目标或末端目标
    必要时通过 IK 转成关节目标

仿真执行层:
    将关节目标送入 Isaac Sim articulation
    由 PhysX 推进机器人与环境交互
```

因此，GenieSim 模拟的是“机器人执行模型 action 后在物理世界中的结果”，而不是完整模拟真实机器人从高层 action 到底层电机驱动的所有控制细节。
