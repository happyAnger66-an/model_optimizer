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
    "native",
    "flashrt",
    "pt_trt_compare",
    "pt_ptq_compare",
    "ptq_trt_compare",
    "pt_ort_compare",
    "pt_flashrt_compare",
]

# 单后端取值（用于分阶段后端矩阵 StagesConfig）。
StageBackend = Literal["pytorch", "tensorrt", "onnxrt", "native", "flashrt"]

# pi05 推理的 5 个可独立替换阶段。
PI05_STAGES: tuple[str, ...] = ("vit", "embed_prefix", "llm", "expert", "denoise")


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
    denoise_adarms_precompute: bool = False
    """denoise 引擎以 AdaRMS 预计算模式导出（输入 ``adarms_mod`` 而非 ``timestep``）时置 True，
    宿主将按步预算 modulation 并喂入（roadmap #22 / docs/optimizer/ddup/adarms_pre_compute.md）。"""
    vit_batch_views: bool = False
    """多视角 batching：把所有相机视角堆成 batch 维一次过 vit 引擎（需 vit 引擎支持动态 batch）。
    与 use_flashrt_siglip_embed_prefix / embed_prefix_engine 互斥。"""


@dataclass
class OnnxRTConfig:
    engine_path: str = ""
    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""
    denoise_adarms_precompute: bool = False
    """同 :attr:`TensorRTConfig.denoise_adarms_precompute`。"""


@dataclass
class FlashRtCalibConfig:
    """FlashRT 静态 FP8 校准参数（详见 docs/optimizer/ddup/flashrt_backend_design.md §6.2）。"""

    n: int = 64
    percentile: float = 99.9
    recalibrate_with_real_data: bool = False


@dataclass
class FlashRtConfig:
    """FlashRT 手写后端配置。"""

    checkpoint_dir: str = ""
    """safetensors 权重源；为空时回退到 tensorrt.engine_path 所在目录。"""
    num_views: int = 2
    use_cuda_graph: bool = True
    lib_dir: str = ""
    """libfmha_*.so 搜索目录（为空则用 flash_rt 默认）。"""
    calib: FlashRtCalibConfig = field(default_factory=FlashRtCalibConfig)


@dataclass
class NativeConfig:
    """Native decoder 运行时配置（Phase A）。"""

    use_cuda_graph: bool = True
    graph_warmup: int = 3
    compile_expert: bool = False
    perf: bool = True


@dataclass
class StagesConfig:
    """分阶段后端矩阵：每个阶段独立选后端，``None`` 表示跟随 ``ServerConfig.mode``。"""

    vit: StageBackend | None = None
    embed_prefix: StageBackend | None = None
    llm: StageBackend | None = None
    expert: StageBackend | None = None
    denoise: StageBackend | None = None


@dataclass
class PTQConfig:
    quant_cfg: str | None = None
    calib_dir: str | None = None
    parts: list[str] = field(default_factory=list)
    measure_quant_error: bool = False
    """若 True，各子模块 PTQ 结束后用校准数据再跑一遍并打印张量级 QDQ 误差。"""


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
    native: NativeConfig = field(default_factory=NativeConfig)
    flashrt: FlashRtConfig = field(default_factory=FlashRtConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    ptq: PTQConfig = field(default_factory=PTQConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)

    def resolve_stages(self) -> dict[str, str]:
        """归一化分阶段后端矩阵：``stages.<stage>`` 优先，否则由 ``mode`` 推导。

        细粒度 ``stages`` 字段覆盖粗粒度 ``mode``。对于纯后端 mode
        （pytorch/tensorrt/onnxrt/flashrt），未显式指定的阶段取该后端；
        对于对比 / PTQ 模式，基线阶段默认 ``pytorch``（第二路策略在 loader 中单独构建）。

        Returns:
            ``{stage: backend}``，stage ∈ :data:`PI05_STAGES`。
        """
        base: str = (
            self.mode
            if self.mode in ("pytorch", "tensorrt", "onnxrt", "native", "flashrt")
            else "pytorch"
        )
        resolved: dict[str, str] = {}
        for stage in PI05_STAGES:
            override = getattr(self.stages, stage, None)
            resolved[stage] = override if override else base
        return resolved

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

        if self.mode == "native":
            # native 模式可纯 PyTorch checkpoint 运行，不需要 engine_path。
            pass

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

        # 分阶段后端矩阵校验
        valid_backends = ("pytorch", "tensorrt", "onnxrt", "native", "flashrt")
        resolved = self.resolve_stages()
        bad_be = {s: b for s, b in resolved.items() if b not in valid_backends}
        if bad_be:
            raise ValueError(
                f"Invalid stage backend(s): {bad_be}; allowed={valid_backends}"
            )

        # 任一阶段走 flashrt（或 mode=flashrt / pt_flashrt_compare）时需要 checkpoint。
        wants_flashrt = "flashrt" in resolved.values() or self.mode in (
            "flashrt",
            "pt_flashrt_compare",
        )
        if wants_flashrt:
            ckpt = self.flashrt.checkpoint_dir or self.tensorrt.engine_path
            if not ckpt:
                raise ValueError(
                    "flashrt backend requires flashrt.checkpoint_dir "
                    "(or tensorrt.engine_path as fallback)"
                )


def _build_nested(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring unknown keys.

    ``from __future__ import annotations`` 使 ``field.type`` 为字符串，故此处
    通过模块 globals 解析前向引用，以支持嵌套 dataclass（如 FlashRtConfig.calib）。
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            continue
        f = fields[key]
        ftype = f.type
        if isinstance(ftype, str):
            ftype = globals().get(ftype)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build_nested(ftype, value)
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
        "native": NativeConfig,
        "flashrt": FlashRtConfig,
        "stages": StagesConfig,
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
