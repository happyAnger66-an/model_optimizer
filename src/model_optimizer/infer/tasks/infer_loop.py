"""run_inference_loop — 主推理控制循环。

对应 record_unified.py 的 ``run_inference_only_loop``：

1. 以固定 fps 采集观测
2. 每 infer_interval 步投递异步推理请求
3. 从 ActionBuffer 取动作下发给机器人
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import InferTaskConfig
from .inference_state import InferenceState
from .policy_client import PolicyInferenceClient

logger = logging.getLogger(__name__)


def run_inference_loop(
    robot: Any,
    config: InferTaskConfig,
    prompt: str,
    stop_event: threading.Event,
    *,
    client: PolicyInferenceClient | None = None,
    state: InferenceState | None = None,
) -> dict[str, Any]:
    """执行纯推理控制循环。

    支持两种使用方式：

    1. **独立模式**（不传 client / state）：自行创建 PolicyInferenceClient 和
       InferenceState，循环结束后释放。适合一次性执行。
    2. **托管模式**（传入 client + state）：复用外部已连接的 client 和正在运行的
       InferenceState（含推理线程）。仅重置 buffer 和队列、执行 warmup，循环结束后
       **不** 关闭 client / 停止推理线程。适合 RobotAgent 等长生命周期宿主。

    Args:
        robot: UnifiedRobot 实例（需提供 ``get_observation()`` 和 ``send_action()``）。
        config: 推理任务配置。
        prompt: 任务描述 prompt。
        stop_event: 外部停止信号。
        client: 已连接的 PolicyInferenceClient（托管模式）。
        state: 已启动的 InferenceState（托管模式）。

    Returns:
        执行摘要 dict：``{"total_steps": int, "elapsed_s": float}``。
    """
    period = 1.0 / config.fps
    owns_resources = client is None  # 独立模式需自行管理资源生命周期

    # 1. 自动从 robot 提取关节名和相机映射
    state_joint_names = config.state_joint_names
    if state_joint_names is None:
        state_joint_names = extract_robot_state_keys(robot)
        logger.info("Auto-extracted %d state joint names from robot", len(state_joint_names))

    if client is None:
        # 独立模式：自行创建
        client = PolicyInferenceClient(
            config=config,
            image_key_map=_build_image_key_map(robot, config),
            state_joint_names=state_joint_names,
        )
        client.connect()

    # warmup 观测 + 推断 action_dim
    obs = robot.get_observation()
    warmup_obs_dict = client.build_obs_dict(obs, prompt)
    warmup_actions = client.get_action(warmup_obs_dict)
    action_dim = warmup_actions.shape[-1]

    if state is None:
        # 独立模式：自行创建 + 启动
        state = InferenceState(config, client, action_dim)
        state.buffer.update_from_inference(
            warmup_actions, start_step=0, current_step=0, enable_fusion=False,
        )
        state.start()
    else:
        # 托管模式：重置（清 buffer、队列），用新 warmup 填充
        state.reset()
        state.buffer.update_from_inference(
            warmup_actions, start_step=0, current_step=0, enable_fusion=False,
        )

    logger.info(
        "Inference loop started: fps=%.1f, action_horizon=%d, "
        "infer_interval=%d, action_dim=%d, state_joints=%d, managed=%s",
        config.fps, config.action_horizon, config.infer_interval,
        action_dim, len(state_joint_names), not owns_resources,
    )

    step_id = 0
    t_start = time.monotonic()

    try:
        while not stop_event.is_set():
            t_loop = time.monotonic()
            state.global_step_id = step_id

            # 采集观测
            obs = robot.get_observation()

            # 每 infer_interval 步投递推理请求
            if step_id % config.infer_interval == 0:
                obs_dict, has_prefix = _build_inference_request(
                    client, obs, prompt, config, state,
                )
                state.submit_request(obs_dict, step_id, has_action_prefix=has_prefix)

            # 非阻塞检查推理结果
            state.check_result()

            # 等待 buffer 非空
            wait_start = time.monotonic()
            while state.buffer.is_empty:
                if stop_event.is_set():
                    break
                time.sleep(0.001)
                if time.monotonic() - wait_start > 3.0:
                    logger.warning("ActionBuffer empty for >3s at step %d", step_id)
                    wait_start = time.monotonic()

            if stop_event.is_set():
                break

            # 取动作并下发
            action = state.buffer.get_next_action()
            _send_action_to_robot(robot, action, state_joint_names)

            step_id += 1

            # busy wait 控频
            _busy_wait_until(t_loop + period)

    except KeyboardInterrupt:
        logger.info("Inference loop interrupted by keyboard")
    finally:
        if owns_resources:
            state.stop()
            client.close()

    elapsed = time.monotonic() - t_start
    logger.info("Inference loop finished: %d steps in %.1f s", step_id, elapsed)
    return {"total_steps": step_id, "elapsed_s": elapsed}


def _build_image_key_map(robot: Any, config: InferTaskConfig) -> dict[str, str]:
    """构建 camera key → obs_dict key 映射。"""
    if config.image_keys is not None:
        # 用户显式配置
        cameras = getattr(robot, "cameras", {})
        cam_names = sorted(cameras.keys())
        mapping = {}
        for cam_name, obs_key in zip(cam_names, config.image_keys):
            mapping[cam_name] = obs_key
        return mapping

    # 自动映射：按名称猜测
    cameras = getattr(robot, "cameras", {})
    mapping: dict[str, str] = {}
    for cam_name in cameras:
        lower = cam_name.lower()
        if "head" in lower or "front" in lower:
            mapping[cam_name] = "observation/front_image"
        elif "left" in lower or "wrist_l" in lower:
            mapping[cam_name] = "observation/wrist_image_lf"
        elif "right" in lower or "wrist" in lower:
            mapping[cam_name] = "observation/wrist_image"
        else:
            mapping[cam_name] = f"observation/{cam_name}"

    return mapping


def _build_inference_request(
    client: PolicyInferenceClient,
    obs: dict,
    prompt: str,
    config: InferTaskConfig,
    state: InferenceState,
) -> tuple[dict, bool]:
    """构造推理请求 obs_dict，含 RTC 字段。

    Returns:
        ``(obs_dict, has_action_prefix)``：obs_dict 和是否携带了 action_prefix。
    """
    action_prefix = None
    action_mask = None
    infer_delay = None
    has_prefix = False

    if config.enable_rtc and not state.buffer.is_empty:
        delay = min(config.default_infer_delay, config.rtc_max_delay)
        prefix, mask, actual_delay = state.buffer.get_future_actions(delay)
        if actual_delay > 0:
            action_prefix = prefix
            action_mask = mask
            infer_delay = actual_delay
            has_prefix = True

    obs_dict = client.build_obs_dict(
        obs, prompt,
        action_prefix=action_prefix,
        action_mask=action_mask,
        infer_delay=infer_delay,
    )
    return obs_dict, has_prefix


def extract_robot_state_keys(robot: Any) -> list[str]:
    """从 UnifiedRobot 自动提取参与推理的关节名列表。

    与原版 ``setup_policy_client`` 逻辑一致：
    从 ``robot._motors_ft`` 中筛选包含 ``"pos"`` 和 ``"joint"`` 的 key，
    去掉 ``.pos`` 后缀得到关节名。

    Args:
        robot: UnifiedRobot 实例。

    Returns:
        关节名列表，如 ``["left_arm_joint_1", "left_arm_joint_2", ...]``。
    """
    motors_ft = getattr(robot, "_motors_ft", None)
    if motors_ft is not None:
        # 原版: [k for k in robot._motors_ft.keys() if "pos" in k and "joint" in k]
        # _motors_ft keys 形如 "left_arm_joint_1.pos"
        state_keys = [k for k in motors_ft.keys() if "pos" in k and "joint" in k]
        # 去掉 .pos 后缀得到关节名
        return [k.removesuffix(".pos") for k in state_keys]

    # fallback: 从 _joint_names 获取（过滤含 "joint" 的）
    joint_names = getattr(robot, "_joint_names", None)
    if joint_names is None:
        fn = getattr(robot, "_get_joint_names", None)
        if fn is not None:
            joint_names = fn()
    if joint_names is not None:
        return [n for n in joint_names if "joint" in n]

    logger.warning("Cannot extract robot state keys; returning empty list")
    return []


def extract_camera_keys(robot: Any) -> list[str]:
    """从 UnifiedRobot 提取相机 key 列表。

    Args:
        robot: UnifiedRobot 实例。

    Returns:
        相机 key 列表，如 ``["head", "left_wrist", "right_wrist"]``。
    """
    cameras = getattr(robot, "cameras", None)
    if cameras is not None:
        return list(cameras.keys())
    camera_cfg = getattr(getattr(robot, "config", None), "cameras", None)
    if camera_cfg is not None:
        return list(camera_cfg.keys())
    return []


def _send_action_to_robot(
    robot: Any, action: Any, state_joint_names: list[str],
) -> None:
    """将 ndarray 动作转为 robot 可接受的 dict 并下发。

    映射方式与原版 ``ActionBuffer.get_next_action`` 一致：
    ``state_joint_names[i]`` → ``{name}.pos`` = action[i]。
    """
    import numpy as np

    action_arr = np.asarray(action, dtype=np.float32)

    if state_joint_names and len(state_joint_names) <= len(action_arr):
        action_dict = {}
        for i, name in enumerate(state_joint_names):
            action_dict[f"{name}.pos"] = float(action_arr[i])
        robot.send_action(action_dict)
    else:
        robot.send_action(action_arr)


def _busy_wait_until(target_time: float) -> None:
    """忙等待到 target_time（比 sleep 更精准）。"""
    while True:
        remaining = target_time - time.monotonic()
        if remaining <= 0:
            break
        if remaining > 0.002:
            time.sleep(remaining * 0.5)
