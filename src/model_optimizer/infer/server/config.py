"""JSON 配置 → dataclass（ServerConfig）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

InferMode = Literal[
    "pytorch",
    "tensorrt",
    "onnxrt",
    "pt_trt_compare",
    "pt_ptq_compare",
    "ptq_trt_compare",
    "pt_ort_compare",
]


@dataclass
class DatasetConfig:
    repo_id: str | None = None
    root: str | None = None
    num_samples: int = 500
    start_index: int = 0


@dataclass
class TensorRTConfig:
    engine_path: str = ""
    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""


@dataclass
class OnnxRTConfig:
    engine_path: str = ""
    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""


@dataclass
class PTQConfig:
    quant_cfg: str | None = None
    calib_dir: str | None = None
    parts: list[str] = field(default_factory=list)


@dataclass
class WebSocketConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws"
    max_fps: float = 0.0
    history_size: int = 0
    gpu_stats_interval: float = 1.0
    jpeg_quality: int = 85
    send_wrist: bool = False
    client_ws_url: str | None = None
    outbound_queue_maxsize: int = 0


@dataclass
class ServeConfig:
    """在线策略推理服务配置。"""

    host: str = "0.0.0.0"
    port: int = 8000
    backend: Literal["local", "remote"] = "local"
    """``local`` 在本地 GPU 加载模型推理；``remote`` 转发到远程 openpi serve_policy。"""

    remote_host: str = "localhost"
    remote_port: int = 8000

    default_prompt: str | None = None
    robot_type: str = "unified_robot"
    unify_action_mode: bool = True

    enable_score: bool = False
    value_temperature: float = 1.0

    record: bool = False
    record_dir: str = "policy_records"


@dataclass
class CalibConfig:
    save_path: str | None = None
    max_samples: int = 0
    item: Literal["all", "vit", "llm", "expert", "denoise", "embed_prefix"] = "all"


@dataclass
class ServerConfig:
    checkpoint: str = ""
    config_name: str = "pi05_libero"
    mode: InferMode = "pytorch"
    device: str | None = None
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    rel_eps: float = 1e-8

    enable_result: bool = True
    """是否启用后处理（metrics/图像编码/StepResult）。

    ``False`` 时推理路径零额外开销：不启动后处理线程，不计算 metrics，
    不编码图像。适用于只关心推理吞吐、校准数据收集等场景。
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)
    onnxrt: OnnxRTConfig = field(default_factory=OnnxRTConfig)
    ptq: PTQConfig = field(default_factory=PTQConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)

    def validate(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint is required")

        if self.mode in ("pt_trt_compare", "ptq_trt_compare"):
            if not self.tensorrt.engine_path:
                raise ValueError(
                    f"mode={self.mode!r} requires tensorrt.engine_path"
                )

        if self.mode == "tensorrt":
            if not self.tensorrt.engine_path:
                raise ValueError(
                    "mode='tensorrt' requires tensorrt.engine_path"
                )

        if self.mode in ("onnxrt", "pt_ort_compare"):
            if not self.onnxrt.engine_path:
                raise ValueError(
                    f"mode={self.mode!r} requires onnxrt.engine_path"
                )

        if self.mode in ("pt_ptq_compare", "ptq_trt_compare"):
            if not self.ptq.quant_cfg:
                raise ValueError(
                    f"mode={self.mode!r} requires ptq.quant_cfg"
                )
            if not self.ptq.calib_dir:
                raise ValueError(
                    f"mode={self.mode!r} requires ptq.calib_dir"
                )
            if not self.ptq.parts:
                raise ValueError(
                    f"mode={self.mode!r} requires non-empty ptq.parts"
                )
            bad = [p for p in self.ptq.parts if p not in ("vit", "llm", "expert", "denoise")]
            if bad:
                raise ValueError(f"Invalid ptq.parts: {bad}")


def _build_nested(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            continue
        f = fields[key]
        if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else None):
            kwargs[key] = _build_nested(f.type, value) if isinstance(value, dict) else value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ServerConfig:
    """Load a ServerConfig from a JSON file."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Config JSON root must be an object, got {type(raw).__name__}")

    nested_fields = {
        "dataset": DatasetConfig,
        "tensorrt": TensorRTConfig,
        "onnxrt": OnnxRTConfig,
        "ptq": PTQConfig,
        "websocket": WebSocketConfig,
        "serve": ServeConfig,
        "calib": CalibConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in nested_fields and isinstance(value, dict):
            kwargs[key] = _build_nested(nested_fields[key], value)
        else:
            kwargs[key] = value

    cfg = ServerConfig(**kwargs)
    cfg.validate()
    return cfg
