"""model_optimizer.infer.tasks — 客户端推理任务框架。

基于 UnifiedRobot + PolicyServer 实现异步推理控制循环，
任务 prompt 通过 ROS2 Action 下发。

核心组件：

- ``InferTaskConfig``：推理任务配置
- ``ActionBuffer``：动作缓冲区（融合、RTC）
- ``PolicyInferenceClient``：策略推理通信客户端
- ``InferenceState``：异步推理会话管理
- ``run_inference_loop``：主推理控制循环
"""

from .action_buffer import ActionBuffer
from .config import InferTaskConfig
from .infer_loop import extract_camera_keys, extract_robot_state_keys, run_inference_loop
from .inference_state import InferenceState
from .policy_client import PolicyInferenceClient

__all__ = [
    "ActionBuffer",
    "InferTaskConfig",
    "InferenceState",
    "PolicyInferenceClient",
    "extract_camera_keys",
    "extract_robot_state_keys",
    "run_inference_loop",
]
