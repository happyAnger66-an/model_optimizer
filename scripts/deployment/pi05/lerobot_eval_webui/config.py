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

    # 单后端：pytorch / tensorrt；compare_mode=True 时忽略此项（固定 PyTorch + TensorRT 双路）
    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    # 双路对比：需 --engine-path 与各 *_engine（同 tensorrt 模式）；与 ptq_compare 互斥
    compare_mode: bool = False

    # PyTorch 浮点 vs 同权重复本上的选择性 PTQ（fake quant）；与 compare_mode 互斥
    ptq_compare: bool = False
    ptq_quant_cfg: Path | None = None
    """ModelOpt 量化配置：``.json`` 或与 ``normalize_quant_cfg`` 一致的 dict；``.py`` 需定义 ``QUANT_CFG``。"""
    ptq_calib_dir: Path | None = None
    """与 ``open_pi05_calib_for_quantize`` 一致：含 ``pi05_{vit,llm,expert}_calib_*`` 的目录。"""
    ptq_parts: tuple[Literal["vit", "llm", "expert"], ...] = dataclasses.field(default_factory=tuple)
    """要量化的子系统，例如 ``--ptq-parts llm expert``。"""
    ptq_layer_report_path: Path | None = None
    """可选：将各 QuantLinear 输出相对 FP 的误差写入该 JSON 路径。"""
    ptq_layer_report_samples: int = 32
    """layer report 使用的连续样本数（自 start_index 起）。"""
    ptq_layer_report_histogram: bool = True
    """layer report 是否附带各层 FP 激活 subsample 直方图（JSON 变大，便于看长尾/异常）。"""
    ptq_layer_report_hist_bins: int = 40
    """直方图 bin 数（线性分箱）。"""
    ptq_layer_report_hist_max_elems: int = 100_000
    """每层每次 forward 参与直方图与统计的最多元素数（降内存/耗时）。"""

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

    仅在 ``inference_mode=pytorch``、``compare_mode`` 或 ``ptq_compare`` 时生效：对每次 ``policy.infer`` 挂 LLM / Expert / ViT 的 forward hook，
    评估结束（或异常退出线程）时在目录下写入各子模块的 ``*_calib_manifest.json`` 与 ``*_calib_shards/`` 分片；
    量化时 ``--calibrate_data`` 传该目录即可流式加载。若仍存在旧的 ``*_calib_datas.pt`` 也会兼容。TensorRT 模式不支持。"""

    gpu_stats_interval_sec: float = 1.0
    """周期向 client 推送 ``type=gpu_stats``（需本机 ``nvidia-smi``）。``0`` 表示关闭。"""

    gpu_device_index: int | None = None
    """``nvidia-smi -i`` 使用的 GPU 下标。``None`` 时从 ``--device`` 解析 ``cuda:N``，否则为 ``0``。"""

    rel_eps: float = 1e-8
    """相对误差分母的 eps：``rel = |pred-gt| / max(|gt|, rel_eps)``。用于屏蔽 gt≈0 导致的发散。"""
