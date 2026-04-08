"""命令行参数（tyro）。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal


@dataclasses.dataclass
class Args:
    checkpoint: Path
    config: str = "pi05_libero"
    num_samples: int = 500
    start_index: int = 0
    dataset_root: Path | None = None
    device: str | None = None

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    engine_path: str = ""
    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""

    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws"
    client_ws_url: str | None = None
    """浏览器默认 WebSocket 地址，写入 ``webui_client/server_hint.json``（页面加载时自动填入）。

    例如远端访问时设置为 ``ws://192.168.1.10:8765/ws``。不填时：若 ``host`` 为 ``0.0.0.0`` / ``::`` 则用
    ``ws://127.0.0.1:{port}{path}``，否则用 ``ws://{host}:{port}{path}``。
    """

    send_wrist: bool = False
    jpeg_quality: int = 85

    max_fps: float = 0.0
    """限制推送帧率（step event/s）。0 表示不限制。"""

    history_size: int = 0
    """缓存最近 N 条消息，新 client 连接后先回放（0 表示不缓存）。"""

    calib_save_path: Path | None = None
    """Pi0.5 校准数据输出目录（与 ``standalone_inference_script.py --calib-save-path`` 相同）。

    仅在 ``inference_mode=pytorch`` 时生效：对每次 ``policy.infer`` 挂 LLM / Expert / ViT 的 forward hook，
    评估结束（或异常退出线程）时在目录下写入各子模块的 ``*_calib_manifest.json`` 与 ``*_calib_shards/`` 分片；
    量化时 ``--calibrate_data`` 传该目录即可流式加载。若仍存在旧的 ``*_calib_datas.pt`` 也会兼容。TensorRT 模式不支持。"""
