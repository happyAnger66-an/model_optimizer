"""RobotAgent — ROS2 Action Server，接收 prompt 驱动推理循环。

外部通过 ROS2 Action Client 下发任务 prompt，RobotAgent 在回调中
驱动 ``run_inference_loop``，支持取消和 feedback 上报。

**持久化资源**：PolicyInferenceClient 和 InferenceState（含推理工作线程）
在 ``__init__`` 时一次性创建，整个 Agent 生命周期内复用。每次收到新 prompt
只做 buffer/queue 重置 + warmup，避免重复建连和加载模型。

用法::

    import rclpy
    from lerobot.robots.unified_robot import UnifiedRobot, UnifiedRobotConfig
    from model_optimizer.infer.tasks import InferTaskConfig
    from model_optimizer.infer.tasks.agents import RobotAgent

    rclpy.init()
    robot = UnifiedRobot(UnifiedRobotConfig())
    robot.connect()

    config = InferTaskConfig(policy_host="localhost", policy_port=8000)
    agent = RobotAgent(robot, config)
    rclpy.spin(agent)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ROS2 Action 类型需要编译 .action 文件后才可用。
# 此处用延迟导入 + 字符串引用，便于在无 ROS2 环境下也能导入本模块做单元测试。
_ACTION_NAME = "infer_task"


class RobotAgent:
    """ROS2 Action Server：接收 prompt，驱动 UnifiedRobot 推理循环。

    PolicyInferenceClient 和 InferenceState 在构造时一次性初始化，
    后续每次 prompt 复用，避免重复加载模型 / 建立连接。

    Args:
        robot: UnifiedRobot 实例（已 connect）。
        config: 推理任务配置。
        node_name: ROS2 节点名。
        action_type: ROS2 Action 类型（编译后的 .action 消息类）。
            为 ``None`` 时使用内置简易 Action 定义。
    """

    def __init__(
        self,
        robot: Any,
        config: Any,
        node_name: str = "robot_agent",
        action_type: Any = None,
    ) -> None:
        import rclpy

        if not rclpy.ok():
            rclpy.init()

        self._node = _RobotAgentNode(
            robot=robot,
            config=config,
            node_name=node_name,
            action_type=action_type,
        )

    @property
    def node(self) -> Any:
        """底层 ROS2 Node，用于 ``rclpy.spin(agent.node)``。"""
        return self._node

    def spin(self) -> None:
        """阻塞式运行（等同于 rclpy.spin(self.node)）。"""
        import rclpy

        try:
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            logger.info("RobotAgent interrupted")
        finally:
            self._node.destroy_node()


class _RobotAgentNode:
    """内部 ROS2 Node 实现。

    将 ROS2 依赖隔离在此类中，便于测试。

    **资源生命周期**：
    - ``__init__``：创建 PolicyInferenceClient → connect → warmup 推断 action_dim
      → 创建 InferenceState → 启动推理工作线程。
    - ``_execute_callback``：每次 prompt 只做 state.reset() + warmup，
      然后运行控制循环（复用已有 client/state）。
    - ``destroy_node``：停止推理线程 → 关闭 client → 销毁 ROS2 Node。
    """

    def __init__(
        self,
        robot: Any,
        config: Any,
        node_name: str,
        action_type: Any,
    ) -> None:
        from rclpy.node import Node
        from rclpy.action import ActionServer
        from ..infer_loop import (
            _build_image_key_map,
            extract_robot_state_keys,
        )
        from ..policy_client import PolicyInferenceClient
        from ..inference_state import InferenceState

        # 动态创建 Node 子类实例
        self._node_impl = type(
            "_AgentNode", (Node,), {}
        )(node_name)

        self._robot = robot
        self._config = config
        self._current_thread: threading.Thread | None = None
        self._current_stop_event: threading.Event | None = None

        # ── 持久化推理资源（一次性初始化）──
        state_joint_names = config.state_joint_names
        if state_joint_names is None:
            state_joint_names = extract_robot_state_keys(robot)
            logger.info("Auto-extracted %d state joint names", len(state_joint_names))
        self._state_joint_names = state_joint_names

        self._client = PolicyInferenceClient(
            config=config,
            image_key_map=_build_image_key_map(robot, config),
            state_joint_names=state_joint_names,
        )
        logger.info("Connecting policy client (backend=%s) …", config.backend)
        self._client.connect()

        # warmup 推断 action_dim
        default_prompt = config.prompt or config.default_prompt or ""
        obs = robot.get_observation()
        warmup_obs = self._client.build_obs_dict(obs, default_prompt)
        warmup_actions = self._client.get_action(warmup_obs)
        action_dim = warmup_actions.shape[-1]
        logger.info("Warmup done, action_dim=%d", action_dim)

        self._state = InferenceState(config, self._client, action_dim)
        self._state.buffer.update_from_inference(
            warmup_actions, start_step=0, current_step=0, enable_fusion=False,
        )
        self._state.start()

        # ── ROS2 Action Server ──
        if action_type is None:
            action_type = _get_default_action_type()

        self._action_type = action_type
        self._action_server = ActionServer(
            self._node_impl,
            action_type,
            _ACTION_NAME,
            execute_callback=self._execute_callback,
        )
        logger.info("RobotAgent action server '%s' ready", _ACTION_NAME)

    def destroy_node(self) -> None:
        # 停止正在运行的控制循环
        if self._current_stop_event is not None:
            self._current_stop_event.set()
        if self._current_thread is not None:
            self._current_thread.join(timeout=5.0)
        # 停止推理线程 + 释放 client
        self._state.stop()
        self._client.close()
        self._node_impl.destroy_node()

    # 为 rclpy.spin() 兼容，代理 Node 接口
    def __getattr__(self, name: str) -> Any:
        return getattr(self._node_impl, name)

    def _execute_callback(self, goal_handle: Any) -> Any:
        """Action Server 执行回调。

        复用持久化的 client 和 state，每次只做 reset + 控制循环。
        """
        from ..infer_loop import run_inference_loop

        prompt = goal_handle.request.prompt
        logger.info("Received infer task: prompt=%r", prompt)

        stop_event = threading.Event()
        self._current_stop_event = stop_event

        result_holder: dict[str, Any] = {}

        def _run() -> None:
            try:
                r = run_inference_loop(
                    self._robot, self._config, prompt, stop_event,
                    client=self._client, state=self._state,
                )
                result_holder.update(r)
            except Exception:
                logger.exception("Inference loop failed")

        infer_thread = threading.Thread(target=_run, daemon=True, name="infer_loop")
        infer_thread.start()
        self._current_thread = infer_thread

        # 定期发布 feedback + 检查取消
        feedback_msg = self._action_type.Feedback()
        while infer_thread.is_alive():
            if goal_handle.is_cancel_requested:
                logger.info("Cancel requested, stopping inference loop")
                stop_event.set()
                goal_handle.canceled()
                infer_thread.join(timeout=5.0)
                return self._make_result(result_holder, canceled=True)

            # 发布 feedback
            feedback_msg.step_id = result_holder.get("total_steps", 0)
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.1)

        infer_thread.join()
        self._current_thread = None
        self._current_stop_event = None

        goal_handle.succeed()
        return self._make_result(result_holder)

    def _make_result(self, result_holder: dict, canceled: bool = False) -> Any:
        result = self._action_type.Result()
        result.success = not canceled
        result.message = "canceled" if canceled else "done"
        result.total_steps = result_holder.get("total_steps", 0)
        return result


def _get_default_action_type() -> Any:
    """尝试导入编译后的 InferTask action 类型。

    如果未编译，抛出 ImportError 并给出提示。
    """
    try:
        from model_optimizer_interfaces.action import InferTask  # type: ignore[import-not-found]
        return InferTask
    except ImportError:
        raise ImportError(
            "ROS2 action type 'InferTask' not found. "
            "Please compile the model_optimizer_interfaces package first, "
            "or pass a custom action_type to RobotAgent().\n"
            "Expected .action definition:\n"
            "  # Goal\n"
            "  string prompt\n"
            "  ---\n"
            "  # Result\n"
            "  bool success\n"
            "  string message\n"
            "  int32 total_steps\n"
            "  ---\n"
            "  # Feedback\n"
            "  int32 step_id\n"
        )
