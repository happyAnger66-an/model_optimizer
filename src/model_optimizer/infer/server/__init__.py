"""model_optimizer.infer.server — Pi0.5 推理服务库。

提供两种使用方式：

1. **纯推理库**（InferServer）::

    from model_optimizer.infer.server import InferServer, load_config

    config = load_config("config.json")
    server = InferServer(config)
    server.load()
    result = server.run_all()
    server.close()

2. **WebSocket 流式服务**（WebSocketInferServer）::

    from model_optimizer.infer.server import WebSocketInferServer, load_config

    config = load_config("config.json")
    ws = WebSocketInferServer(config)
    ws.run()

支持的运行模式（通过 config 中 ``mode`` 字段选择）：

- ``pytorch``：单路 PyTorch 浮点推理
- ``tensorrt``：单路 TensorRT 引擎推理
- ``pt_trt_compare``：PyTorch + TensorRT 双路对比
- ``pt_ptq_compare``：PyTorch + PTQ fake quant 双路对比
- ``ptq_trt_compare``：PTQ + TensorRT 双路对比
"""

from __future__ import annotations

from .config import (
    CalibConfig,
    DatasetConfig,
    PTQConfig,
    ServeConfig,
    ServerConfig,
    TensorRTConfig,
    WebSocketConfig,
    load_config,
)
from .policy_server import PolicyServer
from .result import ChunkPayload, InferResult, StepResult
from .server import InferServer
from .ws_server import WebSocketInferServer

__all__ = [
    "CalibConfig",
    "ChunkPayload",
    "DatasetConfig",
    "InferResult",
    "InferServer",
    "PTQConfig",
    "PolicyServer",
    "ServeConfig",
    "ServerConfig",
    "StepResult",
    "TensorRTConfig",
    "WebSocketConfig",
    "WebSocketInferServer",
    "load_config",
]
