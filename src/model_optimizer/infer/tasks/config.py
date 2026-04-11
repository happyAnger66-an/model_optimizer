"""推理任务配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class InferTaskConfig:
    """客户端推理任务配置，对应 record_unified.py --mode=infer 的参数集。"""

    # ── 推理后端 ──
    backend: Literal["local", "remote"] = "remote"
    """``remote``：通过 WebSocket 连接远程 PolicyServer。
    ``local``：在本进程内加载模型直接推理（需配置 checkpoint / config_name）。"""

    # remote 模式参数
    policy_host: str = "localhost"
    policy_port: int = 8000

    # local 模式参数
    checkpoint: str = ""
    """模型 checkpoint 路径（local 模式必填）。"""
    config_name: str = "pi05_libero"
    """openpi 训练配置名（local 模式）。"""
    device: str | None = None
    """推理设备，如 ``"cuda:0"``。None 时自动选择。"""
    default_prompt: str | None = None
    """本地推理时注入的默认 prompt（可被运行时 prompt 覆盖）。"""
    robot_type: str = "unified_robot"
    """本地推理时的 robot_type 映射。"""

    # ── 控制循环 ──
    fps: float = 30.0
    action_horizon: int = 30
    infer_interval: int = 20

    # ── 动作融合 ──
    fusion_type: Literal["none", "linear", "exponential"] = "linear"
    fusion_start_weight: float = 0.9
    fusion_end_weight: float = 0.1
    fusion_exp_decay: float = 2.0

    # ── RTC (Real-Time Control) ──
    enable_rtc: bool = True
    default_infer_delay: int = 5
    rtc_max_delay: int = 8

    # 默认 prompt（可被 ROS2 Action 覆盖）
    prompt: str = ""

    # warmup
    warmup: bool = True

    # ── 观测字段映射 ──
    image_keys: list[str] | None = None
    """相机 key 映射，如 ``{"cam_head": "observation/front_image", ...}``。
    为 None 时自动探测。"""

    state_joint_names: list[str] | None = None
    """参与 state 拼接的关节名列表。为 None 时使用全部 ``.pos`` 字段。"""
