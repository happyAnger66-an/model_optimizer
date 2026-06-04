"""策略与推理后端加载（PyTorch / TRT / ORT / Native / PTQ）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from termcolor import colored

from .bundle_common import (
    BundleProgress,
    apply_trt_hook_profile_env,
    apply_trt_stage_profile_env,
    create_trained_policy,
    install_compare_pt_stage_perf,
    log_policy_ready,
    trt_trt_second_engine_filenames,
    trt_vit_scale_fix_kw,
)
from .bundle_native import attach_native_executor, load_native_overlay
from .bundle_vit_tap import install_vit_pt_trt_compare_taps
from .config import Args
from .onnxrt_backend import load_onnxrt_engines
from .tensorrt_backend import load_tensorrt_engines


@dataclass
class PolicyBundle:
    policy: Any
    policy_trt: Any | None = None
    policy_ptq: Any | None = None
    native_executor: Any | None = None


def _load_tensorrt_on_policy(
    policy: Any,
    args: Args,
    *,
    engine_path: str,
    vit_engine: str | None = None,
    llm_engine: str | None = None,
    expert_engine: str | None = None,
    denoise_engine: str | None = None,
    embed_prefix_engine: str | None = None,
    vit_batch_views: bool = False,
    trt_perf: bool | None = None,
    trt_perf_warmup: int | None = None,
    trt_perf_print_interval: int | None = None,
    trt_cuda_graph: bool | None = None,
    trt_cuda_graph_warmup: int | None = None,
    denoise_adarms_precompute: bool = False,
) -> None:
    kw: dict[str, Any] = {
        "policy": policy,
        "engine_path": engine_path,
        "precision": args.precision,
        "vit_engine": vit_engine if vit_engine is not None else args.vit_engine,
        "llm_engine": llm_engine if llm_engine is not None else args.llm_engine,
        "expert_engine": expert_engine if expert_engine is not None else args.expert_engine,
        "denoise_engine": denoise_engine if denoise_engine is not None else args.denoise_engine,
        "embed_prefix_engine": embed_prefix_engine
        if embed_prefix_engine is not None
        else args.embed_prefix_engine,
        "trt_vit_scale_fix": trt_vit_scale_fix_kw(args),
    }
    if vit_batch_views:
        kw["vit_batch_views"] = True
    if trt_perf is not None:
        kw["trt_perf"] = trt_perf
    if trt_perf_warmup is not None:
        kw["trt_perf_warmup"] = trt_perf_warmup
    if trt_perf_print_interval is not None:
        kw["trt_perf_print_interval"] = trt_perf_print_interval
    if trt_cuda_graph is not None:
        kw["trt_cuda_graph"] = trt_cuda_graph
    if trt_cuda_graph_warmup is not None:
        kw["trt_cuda_graph_warmup"] = trt_cuda_graph_warmup
    if denoise_adarms_precompute:
        kw["denoise_adarms_precompute"] = True
    load_tensorrt_engines(**kw)


def _load_primary_policy(args: Args, train_cfg: Any, progress: BundleProgress) -> PolicyBundle:
    if getattr(args, "trt_trt_compare", False):
        return _load_trt_trt_dual(args, train_cfg, progress)
    if getattr(args, "trt_ort_compare", False):
        return _load_trt_ort_dual(args, train_cfg, progress)
    return _load_base_pytorch_policy(args, train_cfg, progress)


def _load_trt_trt_dual(args: Args, train_cfg: Any, progress: BundleProgress) -> PolicyBundle:
    if not args.engine_path:
        raise ValueError("trt_trt_compare=True 时必须设置 --engine-path（第一路 TensorRT 引擎目录，如 FP16）。")
    sec = str(getattr(args, "trt_trt_second_engine_path", "") or "").strip()
    if not sec:
        raise ValueError(
            "trt_trt_compare=True 时必须设置 --trt-trt-second-engine-path（第二路引擎目录，如 NVFP4）。"
            "第二路文件名默认与主路相同；若不同请设 --trt-trt-second-llm-engine 等覆盖项。",
        )
    print(colored("[infer] trt_trt_compare：加载第一路 TensorRT（参考）…", "cyan"), flush=True)
    progress.emit("policy_trt_ref", "trt_trt：创建 policy 并挂载第一路 TensorRT …")
    policy = create_trained_policy(train_cfg, args)
    _load_tensorrt_on_policy(policy, args, engine_path=args.engine_path)
    print(colored("[infer] trt_trt_compare：加载第二路 TensorRT …", "cyan"), flush=True)
    progress.emit("policy_trt_target", "trt_trt：创建第二套 policy 并挂载第二路 TensorRT …")
    policy_trt = create_trained_policy(train_cfg, args)
    sec_names = trt_trt_second_engine_filenames(args)
    _load_tensorrt_on_policy(
        policy_trt,
        args,
        engine_path=sec,
        vit_engine=sec_names["vit_engine"],
        llm_engine=sec_names["llm_engine"],
        expert_engine=sec_names["expert_engine"],
        denoise_engine=sec_names["denoise_engine"],
        embed_prefix_engine=sec_names["embed_prefix_engine"],
    )
    print(colored("[infer] trt_trt_compare：双 TensorRT 策略已就绪", "cyan"), flush=True)
    progress.emit("policy_trt_target", "第二路 TensorRT 已挂载（双 TRT 就绪）")
    log_policy_ready(policy, "TRT ref")
    return PolicyBundle(policy=policy, policy_trt=policy_trt)


def _load_trt_ort_dual(args: Args, train_cfg: Any, progress: BundleProgress) -> PolicyBundle:
    if not args.engine_path:
        raise ValueError("trt_ort_compare=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
    ort_ep = getattr(args, "ort_engine_path", "")
    if not ort_ep:
        raise ValueError("trt_ort_compare=True 时必须设置 --ort-engine-path（ONNX 模型目录）。")
    print(colored("[infer] trt_ort_compare：加载 TensorRT 路 policy …", "cyan"), flush=True)
    progress.emit("policy_trt_main", "trt_ort：创建 policy 并挂载 TensorRT …")
    policy = create_trained_policy(train_cfg, args)
    _load_tensorrt_on_policy(policy, args, engine_path=args.engine_path)
    print(colored("[infer] trt_ort_compare：加载 ONNX Runtime 第二路 …", "cyan"), flush=True)
    progress.emit("policy_ort_second", "trt_ort：创建第二套 policy 并挂载 ONNX Runtime …")
    policy_trt = create_trained_policy(train_cfg, args)
    load_onnxrt_engines(
        policy_trt,
        engine_path=ort_ep,
        vit_engine=getattr(args, "ort_vit_engine", ""),
        llm_engine=getattr(args, "ort_llm_engine", ""),
        expert_engine=getattr(args, "ort_expert_engine", ""),
        denoise_engine=getattr(args, "ort_denoise_engine", ""),
        embed_prefix_engine=getattr(args, "ort_embed_prefix_engine", ""),
        ort_providers=tuple(getattr(args, "ort_providers", ())),
    )
    print(colored("[infer] trt_ort_compare：TensorRT + ONNX Runtime 双策略已就绪", "cyan"), flush=True)
    progress.emit("policy_ort_second", "ONNX Runtime 已挂载（TRT vs ORT 双路就绪）")
    log_policy_ready(policy, "TRT")
    return PolicyBundle(policy=policy, policy_trt=policy_trt)


def _load_base_pytorch_policy(args: Args, train_cfg: Any, progress: BundleProgress) -> PolicyBundle:
    print(
        colored("[infer] create_trained_policy（可能较慢，磁盘/显存占用会上升）...", "cyan"),
        flush=True,
    )
    progress.emit("policy_pt", "加载 PyTorch 策略（checkpoint → 内存/显存，可能较慢）…")
    policy = create_trained_policy(train_cfg, args)
    log_policy_ready(policy, "main")
    progress.emit("policy_pt", "PyTorch 策略已就绪")
    return PolicyBundle(policy=policy)


def _validate_ptq_args(args: Args, *, label: str) -> None:
    if args.ptq_quant_cfg is None or not Path(args.ptq_quant_cfg).is_file():
        raise ValueError(f"{label} 需要有效的 --ptq-quant-cfg（存在的 .json 或定义 QUANT_CFG 的 .py）。")
    if args.ptq_calib_dir is None or not Path(args.ptq_calib_dir).expanduser().is_dir():
        raise ValueError(f"{label} 需要 --ptq-calib-dir 指向含 Pi0.5 calib 的目录。")
    if not args.ptq_parts:
        raise ValueError(f"{label} 需要非空 --ptq-parts，例如 vit、llm、expert、denoise。")
    bad = [p for p in args.ptq_parts if p not in ("vit", "llm", "expert", "denoise")]
    if bad:
        raise ValueError(f"非法 --ptq-parts: {bad}（仅允许 vit / llm / expert / denoise）。")


def _attach_secondary_policies(
    args: Args,
    train_cfg: Any,
    bundle: PolicyBundle,
    progress: BundleProgress,
) -> PolicyBundle:
    policy = bundle.policy
    policy_trt = bundle.policy_trt
    policy_ptq = bundle.policy_ptq
    native_executor = bundle.native_executor

    if not getattr(args, "trt_ort_compare", False) and (
        args.compare_mode or getattr(args, "ptq_trt_compare", False)
    ):
        bundle = _attach_compare_or_ptq_trt_second(
            args, train_cfg, policy, progress, native_executor=native_executor
        )
    elif args.ptq_compare:
        bundle = _attach_ptq_second_policy(args, train_cfg, policy, progress)
    elif getattr(args, "ptq_trt_compare", False):
        bundle = _attach_ptq_trt_compare(args, train_cfg, policy, progress)
    elif getattr(args, "ort_compare", False):
        bundle = _attach_ort_compare_second(args, train_cfg, policy, progress)
    elif (
        not getattr(args, "trt_ort_compare", False)
        and not getattr(args, "trt_trt_compare", False)
        and args.inference_mode == "tensorrt"
    ):
        bundle = _attach_tensorrt_single(args, policy, progress)
    elif not getattr(args, "trt_ort_compare", False) and args.inference_mode == "native":
        bundle = _attach_native_single(args, policy, progress)
    elif not getattr(args, "trt_ort_compare", False) and args.inference_mode == "onnxrt":
        bundle = _attach_onnxrt_single(args, policy, progress)

    return bundle


def _attach_compare_or_ptq_trt_second(
    args: Args,
    train_cfg: Any,
    policy: Any,
    progress: BundleProgress,
    *,
    native_executor: Any | None,
) -> PolicyBundle:
    if not args.engine_path:
        raise ValueError("compare_mode=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
    print(colored("[infer] compare_mode：加载第二套 policy 并挂 TensorRT …", "cyan"), flush=True)
    progress.emit("policy_trt", "compare：加载 TensorRT 路 PyTorch 封装并挂载引擎 …")
    apply_trt_stage_profile_env(args)
    apply_trt_hook_profile_env(args, print_on_exit=False)
    policy_trt = create_trained_policy(train_cfg, args)
    compare_warmup = max(int(getattr(args, "perf_profile_warmup_chunks", 10)), 0)
    _load_tensorrt_on_policy(
        policy_trt,
        args,
        engine_path=args.engine_path,
        trt_perf=bool(getattr(args, "trt_perf", True)),
        trt_perf_warmup=compare_warmup,
        trt_perf_print_interval=int(getattr(args, "trt_perf_print_interval", 50)),
        trt_cuda_graph=bool(getattr(args, "trt_cuda_graph", False)),
        trt_cuda_graph_warmup=int(getattr(args, "trt_cuda_graph_warmup", 3)),
        denoise_adarms_precompute=bool(getattr(args, "denoise_adarms_precompute", False)),
    )
    if bool(getattr(args, "native_overlay_on_tensorrt", False)):
        progress.emit("native", "compare：在 TensorRT 路叠加 Native/FlashRT decoder（阶段覆盖）…")
        native_executor = load_native_overlay(
            policy_trt, args, sample_actions_warmup_skips=compare_warmup
        )
        attach_native_executor(policy_trt, native_executor)
        progress.emit("native", "compare：Native/FlashRT 阶段覆盖已生效")
    install_compare_pt_stage_perf(policy, warmup_skips=compare_warmup)
    print(colored("[infer] compare_mode：PyTorch + TensorRT 双策略已就绪", "cyan"), flush=True)
    progress.emit("policy_trt", "TensorRT 引擎已挂载（compare 双路就绪）")
    if getattr(args, "vit_pt_trt_compare", False):
        install_vit_pt_trt_compare_taps(policy, policy_trt, progress)
    return PolicyBundle(
        policy=policy,
        policy_trt=policy_trt,
        native_executor=native_executor,
    )


def _attach_ptq_second_policy(
    args: Args,
    train_cfg: Any,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    _validate_ptq_args(args, label="ptq_compare")
    print(colored("[infer] ptq_compare：第二份 PyTorch policy + 选择性 PTQ …", "cyan"), flush=True)
    progress.emit("ptq_policy", "ptq_compare：加载第二套 PyTorch 策略 …")
    policy_ptq = create_trained_policy(train_cfg, args)
    progress.emit("ptq_policy", "PTQ 路 PyTorch 策略已加载")
    from .ptq_compare import apply_selective_ptq, load_ptq_quant_cfg

    qcfg = load_ptq_quant_cfg(Path(args.ptq_quant_cfg))
    parts_s = ",".join(args.ptq_parts)
    progress.emit("ptq_apply", f"ptq_compare：读取 calib 并对 [{parts_s}] 应用量化（quantize + dynamic）…")
    apply_selective_ptq(
        policy_ptq,
        Path(args.ptq_calib_dir),
        qcfg,
        tuple(args.ptq_parts),
        measure_quant_error=args.ptq_measure_quant_error,
    )
    print(colored("[infer] ptq_compare：浮点 policy + PTQ policy 已就绪", "cyan"), flush=True)
    progress.emit("ptq_apply", "选择性 PTQ 已应用（浮点 + PTQ 双路就绪）")
    return PolicyBundle(policy=policy, policy_ptq=policy_ptq)


def _attach_ptq_trt_compare(
    args: Args,
    train_cfg: Any,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    _validate_ptq_args(args, label="ptq_trt_compare")
    print(colored("[infer] ptq_trt_compare：对 PyTorch policy 应用选择性 PTQ（fake quant）…", "cyan"), flush=True)
    progress.emit("ptq_apply", "ptq_trt_compare：读取 calib 并应用选择性 PTQ（quantize + dynamic）…")
    from .ptq_compare import apply_selective_ptq, load_ptq_quant_cfg

    qcfg = load_ptq_quant_cfg(Path(args.ptq_quant_cfg))
    apply_selective_ptq(
        policy,
        Path(args.ptq_calib_dir),
        qcfg,
        tuple(args.ptq_parts),
        measure_quant_error=args.ptq_measure_quant_error,
    )
    progress.emit("ptq_apply", "PTQ 已应用到主 policy（将作为 pred1）")
    if not args.engine_path:
        raise ValueError("ptq_trt_compare=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
    print(colored("[infer] ptq_trt_compare：加载第二套 policy 并挂 TensorRT …", "cyan"), flush=True)
    progress.emit("policy_trt", "ptq_trt_compare：加载 TensorRT 路 PyTorch 封装并挂载引擎 …")
    policy_trt = create_trained_policy(train_cfg, args)
    _load_tensorrt_on_policy(policy_trt, args, engine_path=args.engine_path)
    progress.emit("policy_trt", "TensorRT 引擎已挂载（PTQ vs TRT 双路就绪）")
    return PolicyBundle(policy=policy, policy_trt=policy_trt)


def _attach_ort_compare_second(
    args: Args,
    train_cfg: Any,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    ort_ep = getattr(args, "ort_engine_path", "")
    if not ort_ep:
        raise ValueError("ort_compare 时必须设置 --ort-engine-path（ONNX 模型目录）。")
    print(colored("[infer] ort_compare：加载第二套 policy 并挂 ONNX Runtime …", "cyan"), flush=True)
    progress.emit("policy_ort", "ort_compare：加载 ONNX Runtime 路 PyTorch 封装并挂载引擎 …")
    policy_trt = create_trained_policy(train_cfg, args)
    load_onnxrt_engines(
        policy_trt,
        engine_path=ort_ep,
        vit_engine=getattr(args, "ort_vit_engine", ""),
        llm_engine=getattr(args, "ort_llm_engine", ""),
        expert_engine=getattr(args, "ort_expert_engine", ""),
        denoise_engine=getattr(args, "ort_denoise_engine", ""),
        embed_prefix_engine=getattr(args, "ort_embed_prefix_engine", ""),
        ort_providers=tuple(getattr(args, "ort_providers", ())),
    )
    print(colored("[infer] ort_compare：PyTorch + ONNX Runtime 双策略已就绪", "cyan"), flush=True)
    progress.emit("policy_ort", "ONNX Runtime 引擎已挂载（compare 双路就绪）")
    return PolicyBundle(policy=policy, policy_trt=policy_trt)


def _attach_tensorrt_single(
    args: Args,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    if not args.engine_path:
        raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
    apply_trt_stage_profile_env(args)
    apply_trt_hook_profile_env(args, print_on_exit=True)
    print(colored("[infer] 加载 TensorRT 引擎 ...", "cyan"), flush=True)
    progress.emit("tensorrt", "加载 TensorRT 引擎（vit/llm/expert 等）…")
    trt_cuda_graph = bool(getattr(args, "trt_cuda_graph", False))
    native_flashrt = bool(getattr(args, "native_flashrt_decoder", False))
    if (
        bool(getattr(args, "native_overlay_on_tensorrt", False))
        and bool(getattr(args, "native_enable_denoise", True))
        and bool(getattr(args, "native_use_cuda_graph", True))
        and not native_flashrt
        and trt_cuda_graph
    ):
        print(
            colored(
                "[infer] native PyTorch denoise CUDA Graph 与 TRT engine CUDA Graph 互斥，"
                "已自动关闭 trt_cuda_graph（vit/llm 仍走 TRT eager launch）",
                "yellow",
            ),
            flush=True,
        )
        trt_cuda_graph = False
    _load_tensorrt_on_policy(
        policy,
        args,
        engine_path=args.engine_path,
        vit_batch_views=bool(getattr(args, "vit_batch_views", False)),
        trt_perf=bool(getattr(args, "trt_perf", True)),
        trt_perf_warmup=int(getattr(args, "trt_perf_warmup", 20)),
        trt_perf_print_interval=int(getattr(args, "trt_perf_print_interval", 50)),
        trt_cuda_graph=trt_cuda_graph,
        trt_cuda_graph_warmup=int(getattr(args, "trt_cuda_graph_warmup", 3)),
        denoise_adarms_precompute=bool(getattr(args, "denoise_adarms_precompute", False)),
    )
    native_executor = None
    if bool(getattr(args, "native_overlay_on_tensorrt", False)):
        progress.emit("native", "在 TensorRT 上叠加 Native decoder（阶段覆盖）…")
        native_executor = load_native_overlay(
            policy,
            args,
            sample_actions_warmup_skips=int(getattr(args, "perf_profile_warmup_chunks", 10)),
        )
        attach_native_executor(policy, native_executor)
        progress.emit("native", "Native 阶段覆盖已生效")
    print(colored("[infer] TensorRT 引擎已就绪", "cyan"), flush=True)
    progress.emit("tensorrt", "TensorRT 引擎已就绪")
    return PolicyBundle(policy=policy, native_executor=native_executor)


def _attach_native_single(
    args: Args,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    print(colored("[infer] 加载 Native decoder 运行时 ...", "cyan"), flush=True)
    progress.emit("native", "加载 Native decoder（expert/denoise）…")
    native_executor = load_native_overlay(
        policy,
        args,
        sample_actions_warmup_skips=int(getattr(args, "perf_profile_warmup_chunks", 10)),
    )
    attach_native_executor(policy, native_executor)
    print(colored("[infer] Native decoder 已就绪", "cyan"), flush=True)
    progress.emit("native", "Native decoder 已就绪")
    return PolicyBundle(policy=policy, native_executor=native_executor)


def _attach_onnxrt_single(
    args: Args,
    policy: Any,
    progress: BundleProgress,
) -> PolicyBundle:
    ort_ep = getattr(args, "ort_engine_path", "")
    if not ort_ep:
        raise ValueError("inference_mode=onnxrt 时必须设置 --ort-engine-path（ONNX 模型目录）。")
    print(colored("[infer] 加载 ONNX Runtime 引擎 ...", "cyan"), flush=True)
    progress.emit("onnxrt", "加载 ONNX Runtime 引擎（vit/llm/expert 等）…")
    load_onnxrt_engines(
        policy,
        engine_path=ort_ep,
        vit_engine=getattr(args, "ort_vit_engine", ""),
        llm_engine=getattr(args, "ort_llm_engine", ""),
        expert_engine=getattr(args, "ort_expert_engine", ""),
        denoise_engine=getattr(args, "ort_denoise_engine", ""),
        embed_prefix_engine=getattr(args, "ort_embed_prefix_engine", ""),
        ort_providers=tuple(getattr(args, "ort_providers", ())),
    )
    print(colored("[infer] ONNX Runtime 引擎已就绪", "cyan"), flush=True)
    progress.emit("onnxrt", "ONNX Runtime 引擎已就绪")
    return PolicyBundle(policy=policy)


def load_policy_bundle(args: Args, train_cfg: Any, progress: BundleProgress) -> PolicyBundle:
    primary = _load_primary_policy(args, train_cfg, progress)
    return _attach_secondary_policies(args, train_cfg, primary, progress)
