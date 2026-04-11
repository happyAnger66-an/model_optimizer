"""PolicyServer — 在线策略推理 WebSocket 服务。

兼容 openpi_client 协议（msgpack 序列化），``record_unified.py`` 的
``WebsocketClientPolicy`` 可零改动直接连接。

支持两种后端：

- **local**：在本地 GPU 加载模型，直接推理。
- **remote**：转发请求到远程 openpi ``serve_policy`` 服务。

用法::

    from model_optimizer.infer.server import PolicyServer, load_config

    config = load_config("serve_config.json")
    server = PolicyServer(config)
    server.load()
    server.run()  # 阻塞，等待客户端连接
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np

from .config import ServerConfig
from ._policy_backend import LocalPolicyBackend, PolicyBackend, RemotePolicyBackend

logger = logging.getLogger(__name__)

ProgressCallback = Any  # Callable[[str, str], None]


def _unpackb_writable(data: bytes) -> Any:
    """反序列化 msgpack 并确保 numpy 数组可写。"""
    from openpi_client import msgpack_numpy

    obj = msgpack_numpy.unpackb(data)

    def _make_writable(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            o = o.copy()
            o.setflags(write=1)
        elif isinstance(o, dict):
            for k, v in o.items():
                o[k] = _make_writable(v)
        elif isinstance(o, (list, tuple)):
            o = type(o)(_make_writable(item) for item in o)
        return o

    return _make_writable(obj)


class PolicyServer:
    """在线策略推理服务，兼容 openpi_client 协议。

    ``load()`` 加载策略后端，``run()`` 启动 WebSocket 服务。
    客户端通过 ``openpi_client.WebsocketClientPolicy`` 连接。
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._backend: PolicyBackend | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, on_progress: ProgressCallback | None = None) -> None:
        """加载策略后端。

        - ``local`` 模式：加载模型到 GPU（复用 policy_loader）。
        - ``remote`` 模式：连接远程 serve_policy 服务。
        """
        if self._loaded:
            raise RuntimeError("PolicyServer already loaded; call close() first")

        if on_progress is None:
            on_progress = lambda stage, msg: None

        serve_cfg = self._config.serve

        if serve_cfg.backend == "local":
            self._backend = self._load_local(on_progress)
        elif serve_cfg.backend == "remote":
            self._backend = self._load_remote(on_progress)
        else:
            raise ValueError(f"Unknown serve backend: {serve_cfg.backend!r}")

        self._loaded = True

    def run(self) -> None:
        """同步阻塞入口：启动 WebSocket 服务。"""
        if not self._loaded:
            raise RuntimeError("Call load() before run()")
        asyncio.run(self.start())

    async def start(self) -> None:
        """异步入口：启动 WebSocket 服务并等待关闭信号。"""
        import websockets.asyncio.server as _server

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

        serve_cfg = self._config.serve
        backend = self._backend

        from openpi_client import msgpack_numpy
        metadata = backend.metadata
        metadata["supports_score_endpoint"] = (
            serve_cfg.enable_score and backend.supports_score
        )

        async def handler(ws: Any) -> None:
            import websockets.exceptions as _wsex
            import websockets.frames

            logger.info("Connection from %s opened", ws.remote_address)
            packer = msgpack_numpy.Packer()

            await ws.send(packer.pack(metadata))

            prev_total_time: float | None = None
            while True:
                try:
                    start_time = time.monotonic()
                    raw = await ws.recv()
                    obs = _unpackb_writable(raw)

                    request_type = obs.pop("_request_type", "infer")
                    if request_type not in ("infer", "score"):
                        raise ValueError(
                            f"Invalid _request_type: {request_type!r}. "
                            f"Must be 'infer' or 'score'."
                        )

                    process_start = time.monotonic()

                    if request_type == "score":
                        if not (serve_cfg.enable_score and backend.supports_score):
                            raise NotImplementedError(
                                "score endpoint not enabled or not supported"
                            )
                        result = backend.score(
                            obs, value_temperature=serve_cfg.value_temperature,
                        )
                    else:
                        result = backend.infer(obs)

                    process_time = time.monotonic() - process_start

                    result["server_timing"] = {"process_ms": process_time * 1000}
                    if prev_total_time is not None:
                        result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await ws.send(packer.pack(result))
                    prev_total_time = time.monotonic() - start_time

                except _wsex.ConnectionClosedOK:
                    logger.info("Connection from %s closed normally", ws.remote_address)
                    break
                except _wsex.ConnectionClosedError:
                    logger.info("Connection from %s closed with error", ws.remote_address)
                    break
                except Exception as exc:
                    logger.error(
                        "Error handling request from %s: %s",
                        ws.remote_address, exc,
                    )
                    logger.debug(traceback.format_exc())
                    try:
                        await ws.send(traceback.format_exc())
                        await ws.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error.",
                        )
                    except Exception:
                        pass
                    break

        def health_check(
            connection: _server.ServerConnection, request: _server.Request,
        ) -> _server.Response | None:
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        logger.info(
            "PolicyServer listening: %s:%s  backend=%s",
            serve_cfg.host, serve_cfg.port, serve_cfg.backend,
        )

        async with _server.serve(
            handler, serve_cfg.host, serve_cfg.port,
            compression=None, max_size=None,
            process_request=health_check,
        ) as server:
            await server.serve_forever()

    def close(self) -> None:
        """释放资源。"""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._loaded = False

    # ── private ──

    def _load_local(self, on_progress: ProgressCallback) -> LocalPolicyBackend:
        from .policy_loader import load_policy_for_serve

        serve_cfg = self._config.serve
        policy = load_policy_for_serve(self._config, on_progress=on_progress)
        return LocalPolicyBackend(
            policy,
            record=serve_cfg.record,
            record_dir=serve_cfg.record_dir,
        )

    def _load_remote(self, on_progress: ProgressCallback) -> RemotePolicyBackend:
        serve_cfg = self._config.serve
        on_progress("remote", f"连接远程策略 {serve_cfg.remote_host}:{serve_cfg.remote_port} …")
        backend = RemotePolicyBackend(serve_cfg.remote_host, serve_cfg.remote_port)
        backend.connect()
        on_progress("remote", "远程策略连接就绪")
        return backend
