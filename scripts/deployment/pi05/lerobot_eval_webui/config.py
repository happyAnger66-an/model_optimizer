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

    # 单后端：pytorch / native / tensorrt / onnxrt；compare_mode=True 时忽略此项（固定 PyTorch + TensorRT 双路）
    inference_mode: Literal["pytorch", "native", "tensorrt", "onnxrt"] = "pytorch"
    # 双路对比：需 --engine-path 与各 *_engine（同 tensorrt 模式）；与 ptq_compare 互斥
    compare_mode: bool = False
    vit_pt_trt_compare: bool = False
    """compare_mode 下额外对比 ViT：PyTorch get_image_features vs TensorRT vit.engine（按 chunk 统计并推送到 WebUI）。"""

    # PyTorch 浮点 vs 同权重复本上的选择性 PTQ（fake quant）；与 compare_mode 互斥
    ptq_compare: bool = False

    # PyTorch PTQ（fake quant） vs TensorRT engine；与 compare_mode / ptq_compare 互斥
    ptq_trt_compare: bool = False

    # PyTorch 浮点 vs ONNX Runtime；与 compare_mode / ptq_compare / ptq_trt_compare / trt_ort_compare 互斥
    ort_compare: bool = False

    # TensorRT engine vs ONNX Runtime；需同时 ``--engine-path`` 与 ``--ort-engine-path``；与其它 compare 互斥
    trt_ort_compare: bool = False

    # 双 TensorRT：主路 ``--engine-path`` 与第二路 ``--trt-trt-second-engine-path``；需 ``--inference-mode tensorrt``；与其它 compare 互斥
    trt_trt_compare: bool = False
    trt_trt_second_engine_path: str = ""
    """第二套 TensorRT 引擎根目录。"""
    # 第二路各子图文件名；留空则沿用主路 ``--vit-engine`` / ``--llm-engine`` 等
    trt_trt_second_vit_engine: str = ""
    trt_trt_second_llm_engine: str = ""
    trt_trt_second_expert_engine: str = ""
    trt_trt_second_denoise_engine: str = ""
    trt_trt_second_embed_prefix_engine: str = ""

    # 以下仅在 ``trt_ort_compare=True`` 时生效：用 Polygraphy 对子图 ONNX 做 TRT vs ORT 逐张量对比（见 ``trt_ort_polygraphy_compare.py``）
    trt_ort_polygraphy_compare: bool = False
    """为 True 时在加载阶段对 vit/llm/expert 等 ONNX 跑一次 Polygraphy 对比，结果写入 meta ``trt_ort_polygraphy``。"""
    trt_ort_polygraphy_mark_all: bool = False
    """为 True 时用 ``MARK_ALL`` 暴露中间张量；必须同时 ``trt_ort_polygraphy_rebuild_trt``（现场编译 TRT，不用预置 .engine）。"""
    trt_ort_polygraphy_rebuild_trt: bool = False
    """与 ``mark_all`` 联用：从 MARK_ALL 后的 ONNX 用 TensorRT builder 编译引擎再与 ORT 对比。"""
    trt_ort_polygraphy_parts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    """非空时仅跑列出的子图，取值 vit / embed_prefix / llm / expert / denoise；空元组表示全部已配置的子图。"""
    trt_ort_polygraphy_ort_providers: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: ("CUDAExecutionProvider", "CPUExecutionProvider")
    )
    trt_ort_polygraphy_max_report_tensors: int = 256
    """每个子图写入 meta 的 tensor 摘要条数上限（按 max_abs 降序）。"""
    trt_ort_polygraphy_seed: int = 0
    """合成输入的 numpy 随机种子（各子图共用基种子，子图内会偏移以避免完全相同）。"""
    ptq_quant_cfg: Path | None = None
    """ModelOpt 量化配置：``.json`` 或与 ``normalize_quant_cfg`` 一致的 dict；``.py`` 需定义 ``QUANT_CFG``。"""
    ptq_calib_dir: Path | None = None
    """与 ``open_pi05_calib_for_quantize`` 一致：含 ``pi05_{vit,llm,expert,denoise}_calib_*`` 的目录。"""
    ptq_parts: tuple[Literal["vit", "llm", "expert", "denoise"], ...] = dataclasses.field(default_factory=tuple)
    """要量化的子系统，例如 ``--ptq-parts llm expert`` 或含 ``denoise``。"""
    ptq_measure_quant_error: bool = False
    """为 True 时，各子模块 PTQ 结束后用校准数据再跑一遍并打印张量级 QDQ 误差。"""
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
    vit_batch_views: bool = False
    """多视角 batching：把所有相机视角堆成 batch 维一次过 vit 引擎（需 vit 引擎按 vit_build_cfg
    的 batch 维 ≥ num_views 编译，且 vit.onnx 含 batch 动态轴）。与 embed_prefix_engine 互斥。"""

    trt_vit_scale_fix: bool = False
    """部分 vit.engine 输出相对 PyTorch 少乘 ``sqrt(hidden_size)``，开启后对 TRT ViT 特征补乘该因子以恢复精度。
    等价于环境变量 ``PI05_TRT_VIT_SCALE_FIX=1``；YAML 显式设置时优先于环境变量。"""

    trt_perf: bool = True
    """打开 TRT engine 的逐次耗时统计（>100 次后打印各 engine 的均值/分位耗时）。"""

    trt_perf_warmup: int = 20
    """TRT engine 性能统计 warmup 次数（前 N 次不计入统计）。"""

    trt_perf_print_interval: int = 50
    """TRT engine 性能统计打印间隔（每 M 次打印一次 mean/p50/p90/p99）。"""

    trt_cuda_graph: bool = False
    """对所有 TRT engine 启用 CUDA Graph 捕获/回放，降低 kernel launch 开销；
    仅在输入 shape 稳定时安全（动态 shape 会触发重捕获）。"""

    trt_cuda_graph_warmup: int = 3
    """CUDA Graph 捕获前的 warmup 次数。"""

    llm_kv_only: bool = False
    """LLM TRT 仅返回/使用 KV 输出；配套 KV-only LLM engine 时可跳过 last_hidden_state。"""

    llm_expected_seq_len: int = 0
    """非 0 时校验 prefix LLM seq_len，帮助固定 shape / CUDA Graph 配置尽早 fail-fast。"""

    denoise_adarms_precompute: bool = False
    """denoise 引擎以 adarms_mod 输入导出时（AdaRMS Dense host 预计算 / roadmap #22），
    在 host 侧预算调制系数并喂入引擎。须与对应的 denoise 引擎（含 adarms_mod 输入）配套。"""

    native_use_cuda_graph: bool = True
    """PyTorch native denoise 启用 CUDA Graph（``denoise_step`` / full-loop V2）。

    **FlashRT denoise（``native_flashrt_decoder``）不使用此项**；FlashRT 走 FVK 整循环，请设 ``false``。"""

    native_full_loop_graph: bool = False
    """PyTorch native 对 sample_actions 全循环做单次 CUDA Graph replay（V2）。FlashRT 路径忽略。"""

    native_graph_warmup: int = 3
    """native 模式 denoise CUDA Graph capture 前 warmup 次数。"""

    native_compile_expert: bool = False
    """native 模式是否对 expert.forward 尝试 torch.compile(reduce-overhead)。"""

    native_enable_expert: bool = True
    native_enable_denoise: bool = True

    native_overlay_on_tensorrt: bool = False
    """当 inference_mode=tensorrt 时，是否叠加 native runtime 覆盖部分阶段（如只覆盖 denoise）。"""

    native_quant_spec_path: str = ""
    """native decoder 量化规格 JSON 路径（Phase B）；为空则不加载。"""

    native_recalib_enable: bool = False
    native_recalib_max_samples: int = 0
    native_recalib_percentile: float = 99.9

    native_flashrt_decoder: bool = False
    """native 模式用仓内 FlashRT decoder（整 10 步循环 + 静态 FP8）替代 denoise loop。
    需在 Thor 上构建 flash_rt_kernels.so（见 scripts/deployment/pi05/build_flashrt_kernels.sh）。"""

    native_flashrt_build_dir: str = ""
    """flash_rt_kernels*.so / libfmha_fp16_strided.so 所在构建目录（不设则用环境变量）。"""

    native_flashrt_fmha_so: str = ""
    """可选：CUTLASS FMHA (libfmha_fp16_strided.so) 路径。"""

    native_flashrt_use_fp8: bool = True
    """FlashRT decoder 是否走静态 FP8 布局；False 走 fp16 baseline。"""

    native_flashrt_act_scales_path: str = ""
    """FlashRT decoder 激活量化 scale 的 JSON 路径（离线量化导出/加载）。"""

    native_flashrt_calibrate: bool = False
    """首次推理跑校准导出 act scales 到 native_flashrt_act_scales_path。"""

    native_flashrt_calib_samples: int = 8
    """标定累计样本数：跨 N 个 observation（KV/noise 各异）对激活 scale 取 max；
    且每个样本内对 10 个扩散步取 max。增大可提升 FP8 标定鲁棒性（减少饱和/掉点）。"""

    trt_release_pytorch_weights: bool = True
    """TRT / FlashRT 挂载成功后释放已被 engine 接管的 PyTorch 子模块权重（vit/llm/expert）。"""

    gpu_mem_profile: bool = False
    """加载阶段打印 PyTorch CUDA 显存分阶段峰值（``[MEM]`` 行）。亦可用环境变量 ``MO_GPU_MEM_PROFILE=1``。"""

    perf_profile_chunk: bool = True
    """打印 chunk 级分解耗时（数据读取/重排/推理/后处理/总计），用于定位 e2e 与引擎时间差。"""

    perf_profile_warmup_chunks: int = 10
    """chunk 级 profile 的 warmup 段数（前 N 段不计）。"""

    perf_profile_print_interval: int = 20
    """chunk 级 profile 打印间隔（每 M 段打印一次）。"""

    perf_profile_top_n: int = 3
    """chunk 级瓶颈排名输出 Top-N（按平均耗时降序）。"""

    trt_enable_stage_profile: bool = True
    """启用 PI0 stage profiler（sample_actions/preprocess/paligemma_forward/denoise_step）。"""

    trt_stage_profile_warmup: int = 10
    """PI0 stage profiler warmup（按 sample_actions 次数跳过）。"""

    trt_enable_hook_profile: bool = True
    """启用 TRT hook 计时（trt.vit/llm/expert/denoise 等入口 wall-time 统计，退出时汇总）。"""

    # ONNX Runtime 引擎（inference_mode=onnxrt 或 ort_compare 时使用）
    ort_engine_path: str = ""
    """ONNX 模型目录（包含 vit.onnx / llm.onnx / expert.onnx 等）。"""
    ort_vit_engine: str = ""
    ort_llm_engine: str = ""
    ort_expert_engine: str = ""
    ort_denoise_engine: str = ""
    ort_embed_prefix_engine: str = ""

    ort_providers: tuple[str, ...] = (
        "TensorRTExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    """ONNX Runtime Execution Provider 顺序（仅保留当前环境 ``ort.get_available_providers()`` 中存在的项）。

    NVFP4（ONNX dtype 23）等通常需要 ``TensorRTExecutionProvider``；若未安装带 TensorRT 的 ORT 构建，将自动跳过该项。
    调试纯 CUDA EP 时可设为 ``--ort-providers CUDAExecutionProvider CPUExecutionProvider``。
    """

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

    outbound_queue_maxsize: int = 0
    """推理线程 → WebSocket 的 Janus 队列容量。``0`` 表示无界；正整数时在下游阻塞时反压推理线程。"""

    history_size: int = 4096
    """缓存最近 N 条出站 JSON（meta/step/done 等），晚连的浏览器可回放；0 表示不缓存。"""

    wait_for_client: bool = False
    """为 True 时：bundle 加载并广播 meta 后，阻塞到首个 WebSocket 客户端连接再开始 chunk 推理。

    避免推理线程在无人订阅时跑完全程（``history_size=0`` 时浏览器会一片空白）。静态页仍需用
    ``python -m http.server`` 等方式打开 ``webui_client/``，勿用 ``file://``。"""

    calib_save_path: Path | None = None
    """Pi0.5 校准数据输出目录（与 ``standalone_inference_script.py --calib-save-path`` 相同）。

    仅在 ``inference_mode=pytorch``、``compare_mode`` 或 ``ptq_compare`` 时生效：对每次 ``policy.infer`` 挂 LLM / Expert / ViT 的 forward hook，
    评估结束（或异常退出线程）时在目录下写入各子模块的 ``*_calib_manifest.json`` 与 ``*_calib_shards/`` 分片；
    量化时 ``--calibrate_data`` 传该目录即可流式加载。若仍存在旧的 ``*_calib_datas.pt`` 也会兼容。TensorRT 模式不支持。"""

    calib_max_samples: int = 0
    """校准数据收集上限（每个 component 单独计数）。0 表示不限制；1 表示每个 component 仅收集 1 条样本。"""

    calib_item: Literal["all", "vit", "llm", "expert", "denoise", "embed_prefix"] = "all"
    """仅收集指定子模型的 calib 数据（默认 all）；``embed_prefix`` 对应 ``PI0Pytorch.embed_prefix`` 输入。"""

    gpu_stats_interval_sec: float = 1.0
    """周期向 client 推送 ``type=gpu_stats``（需本机 ``nvidia-smi``）。``0`` 表示关闭。"""

    gpu_device_index: int | None = None
    """``nvidia-smi -i`` 使用的 GPU 下标。``None`` 时从 ``--device`` 解析 ``cuda:N``，否则为 ``0``。"""

    rel_eps: float = 1e-8
    """相对误差分母的 eps：``rel = |pred-gt| / max(|gt|, rel_eps)``。用于屏蔽 gt≈0 导致的发散。"""

    noise: Literal["random", "fixed"] = "random"
    """流匹配推理初值：``random``（默认）由模型内采样；``fixed`` 时每个数据 chunk 用 ``noise_seed`` 与 chunk 起点 ``idx`` 确定性生成
    ``(action_horizon, action_dim)`` 高斯噪声并传给 ``Policy.infer(..., noise=...)``（形状来自训练配置，与 PyTorch / TensorRT / ONNXRT 后端无关）；多轮可复现；compare / 双路共用同一块噪声。"""

    noise_seed: int = 0
    """仅 ``noise=fixed`` 时与数据集 chunk 起点下标一起决定该 chunk 的初值（``numpy.random.SeedSequence``）。"""
