"""PolicyInferenceClient — 策略推理通信客户端。

支持两种后端：

- **remote**：通过 WebSocket 连接远程 PolicyServer（兼容 openpi 协议）。
- **local**：在本进程内加载模型，直接调用 ``policy.infer()``，零网络开销。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import InferTaskConfig

logger = logging.getLogger(__name__)


class PolicyInferenceClient:
    """策略推理客户端，支持 local / remote 两种后端。

    Args:
        config: 推理任务配置（包含 backend、连接参数、checkpoint 等）。
        image_key_map: UnifiedRobot camera key → obs_dict key 映射，
            如 ``{"head": "observation/front_image", ...}``。
        state_joint_names: 参与 state 拼接的关节名列表（按顺序）。
    """

    def __init__(
        self,
        config: InferTaskConfig,
        image_key_map: dict[str, str] | None = None,
        state_joint_names: list[str] | None = None,
    ) -> None:
        self._config = config
        self._image_key_map = image_key_map or {}
        self._state_joint_names = state_joint_names

        # 统一的推理接口：infer(obs) -> dict
        self._infer_fn: Any = None
        self._metadata: dict[str, Any] = {}
        self._backend_obj: Any = None

    def connect(self) -> dict[str, Any]:
        """初始化后端（连接远程服务 或 加载本地模型）。

        Returns:
            服务端/模型 metadata。
        """
        if self._config.backend == "remote":
            return self._connect_remote()
        elif self._config.backend == "local":
            return self._connect_local()
        else:
            raise ValueError(f"Unknown backend: {self._config.backend!r}")

    @property
    def is_connected(self) -> bool:
        return self._infer_fn is not None

    @property
    def server_metadata(self) -> dict[str, Any]:
        return self._metadata

    def build_obs_dict(
        self,
        observation: dict[str, Any],
        prompt: str,
        action_prefix: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        infer_delay: int | None = None,
    ) -> dict[str, Any]:
        """从 UnifiedRobot observation 构造推理 obs_dict。

        Args:
            observation: ``UnifiedRobot.get_observation()`` 返回的 dict，
                包含 ``{cam_key: ndarray(HWC), joint.pos: float, ...}``。
            prompt: 自然语言任务描述。
            action_prefix: RTC 已执行动作前缀 (可选)。
            action_mask: RTC 前缀掩码 (可选)。
            infer_delay: RTC 延迟步数 (可选)。

        Returns:
            策略推理所需的 obs_dict。
        """
        obs_dict: dict[str, Any] = {}

        # 1. 图像：HWC → CHW
        for cam_key, obs_key in self._image_key_map.items():
            if cam_key in observation:
                img = np.asarray(observation[cam_key])
                if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
                obs_dict[obs_key] = img

        # 2. 关节状态 → float32 数组
        state = self._extract_state(observation)
        obs_dict["observation/state"] = state

        # 3. prompt
        obs_dict["prompt"] = prompt

        # 4. RTC 字段
        if action_prefix is not None:
            obs_dict["action"] = action_prefix
        if action_mask is not None:
            obs_dict["action_mask"] = action_mask
        if infer_delay is not None:
            obs_dict["infer_delay"] = infer_delay

        return obs_dict

    def get_action(self, obs_dict: dict[str, Any]) -> np.ndarray:
        """调用策略推理，返回 actions。

        Args:
            obs_dict: 已构造好的观测字典。

        Returns:
            ``np.ndarray(action_horizon, action_dim)``。
        """
        if self._infer_fn is None:
            raise RuntimeError("PolicyInferenceClient not connected; call connect() first")

        result = self._infer_fn(obs_dict)
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions

    def close(self) -> None:
        """释放资源。"""
        if self._backend_obj is not None:
            close_fn = getattr(self._backend_obj, "close", None)
            if close_fn is not None:
                close_fn()
            self._backend_obj = None
        self._infer_fn = None
        self._metadata = {}

    # ── private ──

    def _connect_remote(self) -> dict[str, Any]:
        """WebSocket 远程连接。"""
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        cfg = self._config
        logger.info("Connecting to remote policy at %s:%s …", cfg.policy_host, cfg.policy_port)
        client = WebsocketClientPolicy(host=cfg.policy_host, port=cfg.policy_port)
        self._metadata = client.get_server_metadata()
        self._infer_fn = client.infer
        self._backend_obj = client
        logger.info("Remote policy connected, metadata keys: %s", list(self._metadata.keys()))
        return self._metadata

    def _connect_local(self) -> dict[str, Any]:
        """本地 GPU 加载模型。"""
        cfg = self._config
        if not cfg.checkpoint:
            raise ValueError("backend='local' requires config.checkpoint")

        from model_optimizer.infer.server.config import ServerConfig, ServeConfig
        from model_optimizer.infer.server.policy_loader import load_policy_for_serve
        from model_optimizer.infer.server._policy_backend import LocalPolicyBackend

        logger.info(
            "Loading local policy: checkpoint=%s, config=%s, device=%s",
            cfg.checkpoint, cfg.config_name, cfg.device,
        )

        # 构造 ServerConfig 用于复用 load_policy_for_serve
        serve_cfg = ServeConfig(
            default_prompt=cfg.default_prompt,
            robot_type=cfg.robot_type,
        )
        server_cfg = ServerConfig(
            checkpoint=cfg.checkpoint,
            config_name=cfg.config_name,
            device=cfg.device,
            serve=serve_cfg,
        )

        policy = load_policy_for_serve(
            server_cfg,
            on_progress=lambda stage, msg: logger.info("[%s] %s", stage, msg),
        )

        backend = LocalPolicyBackend(policy)
        self._metadata = backend.metadata
        self._infer_fn = backend.infer
        self._backend_obj = backend
        logger.info("Local policy loaded, metadata keys: %s", list(self._metadata.keys()))
        return self._metadata

    def _extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        """从 observation 提取关节状态向量。"""
        if self._state_joint_names is not None:
            values = []
            for name in self._state_joint_names:
                key = f"{name}.pos"
                values.append(float(observation.get(key, 0.0)))
            return np.array(values, dtype=np.float32)

        # 自动提取所有 .pos 字段（保持排序确定性）
        pos_keys = sorted(k for k in observation if k.endswith(".pos"))
        values = [float(observation[k]) for k in pos_keys]
        return np.array(values, dtype=np.float32)
