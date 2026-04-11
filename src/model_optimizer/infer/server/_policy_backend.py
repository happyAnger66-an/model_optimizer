"""策略推理后端：本地 GPU 推理 / 远程转发。"""

from __future__ import annotations

import logging
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import ServerConfig

logger = logging.getLogger(__name__)


class PolicyBackend(ABC):
    """在线策略推理后端接口。"""

    @abstractmethod
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """执行推理，返回包含 ``actions`` 的结果字典。"""
        ...

    def score(self, obs: dict[str, Any], *, value_temperature: float = 1.0) -> dict[str, Any]:
        """计算 value score（可选，默认不支持）。"""
        raise NotImplementedError("score not supported by this backend")

    @property
    def supports_score(self) -> bool:
        return False

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """策略元信息，连接建立时发送给客户端。"""
        ...

    def close(self) -> None:
        """释放资源。"""


class LocalPolicyBackend(PolicyBackend):
    """本地 GPU 推理，直接调 openpi Policy.infer()。"""

    def __init__(self, policy: Any, *, record: bool = False, record_dir: str = "policy_records") -> None:
        self._policy = policy
        self._supports_score = hasattr(policy, "score_observation")
        self._metadata: dict[str, Any] = getattr(policy, "metadata", None) or {}

        if record:
            try:
                from openpi.policies.policy import PolicyRecorder
                self._policy = PolicyRecorder(policy, record_dir)
                logger.info("PolicyRecorder enabled → %s", record_dir)
            except ImportError:
                logger.warning("openpi PolicyRecorder not available, recording disabled")

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._policy.infer(obs)

    def score(self, obs: dict[str, Any], *, value_temperature: float = 1.0) -> dict[str, Any]:
        if not self._supports_score:
            raise NotImplementedError("score not supported: model has no RL value head")
        return self._policy.score_observation(obs, value_temperature=value_temperature)

    @property
    def supports_score(self) -> bool:
        return self._supports_score

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


class RemotePolicyBackend(PolicyBackend):
    """转发到远程 openpi serve_policy WebSocket 服务。"""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._client: Any = None
        self._server_metadata: dict[str, Any] = {}

    def connect(self) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        logger.info("Connecting to remote policy at %s:%s …", self._host, self._port)
        self._client = WebsocketClientPolicy(host=self._host, port=self._port)
        self._server_metadata = self._client.get_server_metadata()
        logger.info("Remote policy connected, metadata: %s", list(self._server_metadata.keys()))

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("RemotePolicyBackend not connected; call connect() first")
        return self._client.infer(obs)

    def score(self, obs: dict[str, Any], *, value_temperature: float = 1.0) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("RemotePolicyBackend not connected; call connect() first")
        return self._client.score_observation(obs)

    @property
    def supports_score(self) -> bool:
        return self._server_metadata.get("supports_score_endpoint", False)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def close(self) -> None:
        self._client = None
