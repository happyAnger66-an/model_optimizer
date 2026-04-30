"""加载数据集、策略与 meta 消息（推理线程内调用）。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openpi.policies import policy_config
from openpi.training import config as _config
from termcolor import colored

from .calib import start_pi05_calib_collectors
from .config import Args
from .dataset import (
    build_repack_only,
    global_episode_id_per_frame,
    make_lerobot_dataset,
    unwrap_lerobot_base,
)
from .gpu_stats import effective_gpu_index
from .protocol import event_to_json
from .running_stats import (
    RunningErrorStats,
    RunningPerDimMsePctStats,
    RunningPerDimPairMseStats,
    RunningPerDimRelP99Stats,
    RunningVitCompareStats,
)
from .onnxrt_backend import load_onnxrt_engines
from .tensorrt_backend import load_tensorrt_engines


def _trt_trt_second_engine_filenames(args: Args) -> dict[str, str]:
    """第二路引擎文件名：``trt_trt_second_*`` 非空优先，否则回退到主路 ``*_engine``。"""

    def _coalesce(second: str, primary: str) -> str:
        s = (second or "").strip()
        return s if s else (primary or "")

    return {
        "vit_engine": _coalesce(str(getattr(args, "trt_trt_second_vit_engine", "")), args.vit_engine),
        "llm_engine": _coalesce(str(getattr(args, "trt_trt_second_llm_engine", "")), args.llm_engine),
        "expert_engine": _coalesce(str(getattr(args, "trt_trt_second_expert_engine", "")), args.expert_engine),
        "denoise_engine": _coalesce(str(getattr(args, "trt_trt_second_denoise_engine", "")), args.denoise_engine),
        "embed_prefix_engine": _coalesce(
            str(getattr(args, "trt_trt_second_embed_prefix_engine", "")),
            args.embed_prefix_engine,
        ),
    }


def load_infer_bundle(
    args: Args,
    run_id: str,
    *,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """在专用推理线程中执行：数据集 + policy + TRT，避免阻塞 asyncio 事件循环。

    on_progress(stage_id, message)：可选，供 WebUI 推送加载步骤（stage_id 稳定、便于 client 去重/排序）。
    """

    def _p(stage: str, msg: str) -> None:
        if on_progress is not None:
            on_progress(stage, msg)

    modes = [
        bool(args.compare_mode),
        bool(args.ptq_compare),
        bool(getattr(args, "ptq_trt_compare", False)),
        bool(getattr(args, "ort_compare", False)),
        bool(getattr(args, "trt_ort_compare", False)),
        bool(getattr(args, "trt_trt_compare", False)),
    ]
    if sum(1 for x in modes if x) > 1:
        raise ValueError(
            "compare_mode / ptq_compare / ptq_trt_compare / ort_compare / trt_ort_compare / trt_trt_compare 互斥，请勿同时开启。"
        )
    if (args.ptq_compare or getattr(args, "ptq_trt_compare", False)) and args.inference_mode != "pytorch":
        raise ValueError("ptq_compare / ptq_trt_compare 仅支持 inference_mode=pytorch。")
    if getattr(args, "trt_trt_compare", False) and args.inference_mode != "tensorrt":
        raise ValueError("trt_trt_compare=True 时必须设置 --inference-mode tensorrt（双路均为 TensorRT 引擎）。")

    print(colored("[infer] get_config + data_config ...", "cyan"), flush=True)
    _p("config", "读取训练配置与 data_config …")
    train_cfg = _config.get_config(args.config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_dim = int(getattr(train_cfg.model, "action_dim", 0) or 0)
    if action_dim <= 0:
        raise ValueError("train_cfg.model 缺少有效的 action_dim，无法构建流匹配噪声形状。")
    action_keys = tuple(data_config.action_sequence_keys)

    print(colored(f"[infer] LeRobotDataset(repo={data_config.repo_id!r}) ...", "cyan"), flush=True)
    _p("dataset", f"加载 LeRobot 数据集（repo_id={data_config.repo_id}）…")
    dataset = make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = build_repack_only(data_config)
    _p("dataset", f"数据集就绪（共 {len(dataset)} 条）")

    policy_trt: Any | None = None
    policy_ptq: Any | None = None

    if getattr(args, "trt_trt_compare", False):
        if not args.engine_path:
            raise ValueError("trt_trt_compare=True 时必须设置 --engine-path（第一路 TensorRT 引擎目录，如 FP16）。")
        sec = str(getattr(args, "trt_trt_second_engine_path", "") or "").strip()
        if not sec:
            raise ValueError(
                "trt_trt_compare=True 时必须设置 --trt-trt-second-engine-path（第二路引擎目录，如 NVFP4）。"
                "第二路文件名默认与主路相同；若不同请设 --trt-trt-second-llm-engine 等覆盖项。",
            )
        print(colored("[infer] trt_trt_compare：加载第一路 TensorRT（参考）…", "cyan"), flush=True)
        _p("policy_trt_ref", "trt_trt：创建 policy 并挂载第一路 TensorRT …")
        policy = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        load_tensorrt_engines(
            policy,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] trt_trt_compare：加载第二路 TensorRT …", "cyan"), flush=True)
        _p("policy_trt_target", "trt_trt：创建第二套 policy 并挂载第二路 TensorRT …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        sec_names = _trt_trt_second_engine_filenames(args)
        load_tensorrt_engines(
            policy_trt,
            engine_path=sec,
            precision=args.precision,
            vit_engine=sec_names["vit_engine"],
            llm_engine=sec_names["llm_engine"],
            expert_engine=sec_names["expert_engine"],
            denoise_engine=sec_names["denoise_engine"],
            embed_prefix_engine=sec_names["embed_prefix_engine"],
        )
        print(colored("[infer] trt_trt_compare：双 TensorRT 策略已就绪", "cyan"), flush=True)
        _p("policy_trt_target", "第二路 TensorRT 已挂载（双 TRT 就绪）")
        try:
            pd = getattr(policy, "_pytorch_device", None)
            is_pt = getattr(policy, "_is_pytorch_model", None)
            print(
                colored(f"[infer] policy(TRT ref) 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
                flush=True,
            )
        except Exception:
            pass
    elif getattr(args, "trt_ort_compare", False):
        if not args.engine_path:
            raise ValueError("trt_ort_compare=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
        ort_ep = getattr(args, "ort_engine_path", "")
        if not ort_ep:
            raise ValueError("trt_ort_compare=True 时必须设置 --ort-engine-path（ONNX 模型目录）。")
        print(colored("[infer] trt_ort_compare：加载 TensorRT 路 policy …", "cyan"), flush=True)
        _p("policy_trt_main", "trt_ort：创建 policy 并挂载 TensorRT …")
        policy = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        load_tensorrt_engines(
            policy,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] trt_ort_compare：加载 ONNX Runtime 第二路 …", "cyan"), flush=True)
        _p("policy_ort_second", "trt_ort：创建第二套 policy 并挂载 ONNX Runtime …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
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
        _p("policy_ort_second", "ONNX Runtime 已挂载（TRT vs ORT 双路就绪）")
        try:
            pd = getattr(policy, "_pytorch_device", None)
            is_pt = getattr(policy, "_is_pytorch_model", None)
            print(
                colored(f"[infer] policy(TRT) 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
                flush=True,
            )
        except Exception:
            pass
    else:
        print(colored("[infer] create_trained_policy（可能较慢，磁盘/显存占用会上升）...", "cyan"), flush=True)
        _p("policy_pt", "加载 PyTorch 策略（checkpoint → 内存/显存，可能较慢）…")
        policy = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        try:
            pd = getattr(policy, "_pytorch_device", None)
            is_pt = getattr(policy, "_is_pytorch_model", None)
            print(
                colored(f"[infer] policy 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
                flush=True,
            )
        except Exception:
            pass
        _p("policy_pt", "PyTorch 策略已就绪")

    if not getattr(args, "trt_ort_compare", False) and (
        args.compare_mode or getattr(args, "ptq_trt_compare", False)
    ):
        if not args.engine_path:
            raise ValueError("compare_mode=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
        print(colored("[infer] compare_mode：加载第二套 policy 并挂 TensorRT …", "cyan"), flush=True)
        _p("policy_trt", "compare：加载 TensorRT 路 PyTorch 封装并挂载引擎 …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        load_tensorrt_engines(
            policy_trt,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] compare_mode：PyTorch + TensorRT 双策略已就绪", "cyan"), flush=True)
        _p("policy_trt", "TensorRT 引擎已挂载（compare 双路就绪）")

        if getattr(args, "vit_pt_trt_compare", False):
            # 在 TRT 替换 get_image_features 之后挂 tap：同一输入下抓 PT vs TRT 的 ViT 输出差异（按 chunk）。
            def _policy_torch_model(pol: Any) -> Any:
                m = getattr(pol, "_model", None)
                if m is not None:
                    return m
                inner = getattr(pol, "_policy", None)
                if inner is not None:
                    return getattr(inner, "_model", None)
                return None

            def _install_vit_tap(pol: Any, tag: str) -> None:
                mdl = _policy_torch_model(pol)
                if mdl is None:
                    raise RuntimeError(f"vit_pt_trt_compare: cannot resolve policy torch model for {tag}")
                mm = mdl.paligemma_with_expert.paligemma.model
                orig = getattr(mm, "get_image_features", None)
                if not callable(orig):
                    raise RuntimeError(f"vit_pt_trt_compare: {tag} missing callable get_image_features")

                def _tensor_stats(x: Any) -> dict[str, Any] | None:
                    try:
                        import torch

                        if not torch.is_tensor(x):
                            return None
                        t = x.detach()
                        # Use float32 for stable statistics; keep it cheap (no flatten copy).
                        tf = t.to(torch.float32)
                        return {
                            "shape": list(t.shape),
                            "dtype": str(t.dtype),
                            "min": float(tf.amin().item()),
                            "max": float(tf.amax().item()),
                            "mean": float(tf.mean().item()),
                            "std": float(tf.std(unbiased=False).item()),
                        }
                    except Exception:
                        return None

                def wrapped(pixel_values, *a, **kw):
                    out = orig(pixel_values, *a, **kw)
                    try:
                        setattr(mm, f"_webui_last_vit_in_{tag}", pixel_values.detach())
                    except Exception:
                        setattr(mm, f"_webui_last_vit_in_{tag}", pixel_values)
                    setattr(mm, f"_webui_last_vit_in_stats_{tag}", _tensor_stats(pixel_values))
                    try:
                        setattr(mm, f"_webui_last_vit_out_{tag}", out.detach())
                    except Exception:
                        setattr(mm, f"_webui_last_vit_out_{tag}", out)
                    return out

                mm.get_image_features = wrapped

                def fetch_and_clear():
                    x = getattr(mm, f"_webui_last_vit_in_{tag}", None)
                    y = getattr(mm, f"_webui_last_vit_out_{tag}", None)
                    s = getattr(mm, f"_webui_last_vit_in_stats_{tag}", None)
                    for nm in (
                        f"_webui_last_vit_in_{tag}",
                        f"_webui_last_vit_out_{tag}",
                        f"_webui_last_vit_in_stats_{tag}",
                    ):
                        if hasattr(mm, nm):
                            try:
                                delattr(mm, nm)
                            except Exception:
                                pass
                    return x, y, s

                setattr(pol, f"_webui_fetch_vit_io_{tag}", fetch_and_clear)

            _p("vit_compare", "compare_mode：安装 ViT get_image_features tap（PT vs TRT）…")
            _install_vit_tap(policy, "pt")
            _install_vit_tap(policy_trt, "trt")
            _p("vit_compare", "ViT tap 已安装（将按 chunk 推送 PT↔TRT 摘要）")
    elif args.ptq_compare:
        if args.ptq_quant_cfg is None or not Path(args.ptq_quant_cfg).is_file():
            raise ValueError("ptq_compare 需要有效的 --ptq-quant-cfg（存在的 .json 或定义 QUANT_CFG 的 .py）。")
        if args.ptq_calib_dir is None or not Path(args.ptq_calib_dir).expanduser().is_dir():
            raise ValueError("ptq_compare 需要 --ptq-calib-dir 指向含 Pi0.5 calib 的目录。")
        if not args.ptq_parts:
            raise ValueError("ptq_compare 需要非空 --ptq-parts，例如 vit、llm、expert、denoise。")
        bad = [p for p in args.ptq_parts if p not in ("vit", "llm", "expert", "denoise")]
        if bad:
            raise ValueError(f"非法 --ptq-parts: {bad}（仅允许 vit / llm / expert / denoise）。")

        print(colored("[infer] ptq_compare：第二份 PyTorch policy + 选择性 PTQ …", "cyan"), flush=True)
        _p("ptq_policy", "ptq_compare：加载第二套 PyTorch 策略 …")
        policy_ptq = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        _p("ptq_policy", "PTQ 路 PyTorch 策略已加载")
        from .ptq_compare import apply_selective_ptq, load_ptq_quant_cfg

        qcfg = load_ptq_quant_cfg(Path(args.ptq_quant_cfg))
        parts_s = ",".join(args.ptq_parts)
        _p("ptq_apply", f"ptq_compare：读取 calib 并对 [{parts_s}] 应用量化（quantize + dynamic）…")
        apply_selective_ptq(
            policy_ptq,
            Path(args.ptq_calib_dir),
            qcfg,
            tuple(args.ptq_parts),
            measure_quant_error=args.ptq_measure_quant_error,
        )

        print(colored("[infer] ptq_compare：浮点 policy + PTQ policy 已就绪", "cyan"), flush=True)
        _p("ptq_apply", "选择性 PTQ 已应用（浮点 + PTQ 双路就绪）")
    elif getattr(args, "ptq_trt_compare", False):
        # PTQ(Pytorch) vs TRT(engine)
        if args.ptq_quant_cfg is None or not Path(args.ptq_quant_cfg).is_file():
            raise ValueError("ptq_trt_compare 需要有效的 --ptq-quant-cfg（存在的 .json 或定义 QUANT_CFG 的 .py）。")
        if args.ptq_calib_dir is None or not Path(args.ptq_calib_dir).expanduser().is_dir():
            raise ValueError("ptq_trt_compare 需要 --ptq-calib-dir 指向含 Pi0.5 calib 的目录。")
        if not args.ptq_parts:
            raise ValueError("ptq_trt_compare 需要非空 --ptq-parts，例如 vit、llm、expert、denoise。")
        bad = [p for p in args.ptq_parts if p not in ("vit", "llm", "expert", "denoise")]
        if bad:
            raise ValueError(f"非法 --ptq-parts: {bad}（仅允许 vit / llm / expert / denoise）。")

        print(colored("[infer] ptq_trt_compare：对 PyTorch policy 应用选择性 PTQ（fake quant）…", "cyan"), flush=True)
        _p("ptq_apply", "ptq_trt_compare：读取 calib 并应用选择性 PTQ（quantize + dynamic）…")
        from .ptq_compare import apply_selective_ptq, load_ptq_quant_cfg

        qcfg = load_ptq_quant_cfg(Path(args.ptq_quant_cfg))
        apply_selective_ptq(
            policy,
            Path(args.ptq_calib_dir),
            qcfg,
            tuple(args.ptq_parts),
            measure_quant_error=args.ptq_measure_quant_error,
        )
        _p("ptq_apply", "PTQ 已应用到主 policy（将作为 pred1）")

        if not args.engine_path:
            raise ValueError("ptq_trt_compare=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
        print(colored("[infer] ptq_trt_compare：加载第二套 policy 并挂 TensorRT …", "cyan"), flush=True)
        _p("policy_trt", "ptq_trt_compare：加载 TensorRT 路 PyTorch 封装并挂载引擎 …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        load_tensorrt_engines(
            policy_trt,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        _p("policy_trt", "TensorRT 引擎已挂载（PTQ vs TRT 双路就绪）")
    elif getattr(args, "ort_compare", False):
        ort_ep = getattr(args, "ort_engine_path", "")
        if not ort_ep:
            raise ValueError("ort_compare 时必须设置 --ort-engine-path（ONNX 模型目录）。")
        print(colored("[infer] ort_compare：加载第二套 policy 并挂 ONNX Runtime …", "cyan"), flush=True)
        _p("policy_ort", "ort_compare：加载 ONNX Runtime 路 PyTorch 封装并挂载引擎 …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
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
        _p("policy_ort", "ONNX Runtime 引擎已挂载（compare 双路就绪）")
    elif (
        not getattr(args, "trt_ort_compare", False)
        and not getattr(args, "trt_trt_compare", False)
        and args.inference_mode == "tensorrt"
    ):
        if not args.engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
        print(colored("[infer] 加载 TensorRT 引擎 ...", "cyan"), flush=True)
        _p("tensorrt", "加载 TensorRT 引擎（vit/llm/expert 等）…")
        load_tensorrt_engines(
            policy,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] TensorRT 引擎已就绪", "cyan"), flush=True)
        _p("tensorrt", "TensorRT 引擎已就绪")
    elif not getattr(args, "trt_ort_compare", False) and args.inference_mode == "onnxrt":
        ort_ep = getattr(args, "ort_engine_path", "")
        if not ort_ep:
            raise ValueError("inference_mode=onnxrt 时必须设置 --ort-engine-path（ONNX 模型目录）。")
        print(colored("[infer] 加载 ONNX Runtime 引擎 ...", "cyan"), flush=True)
        _p("onnxrt", "加载 ONNX Runtime 引擎（vit/llm/expert 等）…")
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
        _p("onnxrt", "ONNX Runtime 引擎已就绪")

    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    base_ds = unwrap_lerobot_base(dataset)
    ep_per_frame = global_episode_id_per_frame(base_ds, n)

    ptq_layer_report_path_resolved: Path | None = None
    ptq_layer_report_data: dict[str, Any] | None = None
    if args.ptq_compare and args.ptq_layer_report_path is not None:
        from .ptq_compare import write_ptq_layer_report

        ptq_layer_report_path_resolved = Path(args.ptq_layer_report_path).expanduser().resolve()
        _p("ptq_report", "生成分层 PTQ 报告（hook 对比 FP/PTQ，可能较慢）…")
        write_ptq_layer_report(
            policy,
            policy_ptq,
            tuple(args.ptq_parts),
            dataset=dataset,
            repack_fn=repack_fn,
            start_index=int(args.start_index),
            num_samples=int(args.ptq_layer_report_samples),
            report_path=ptq_layer_report_path_resolved,
            include_activation_histogram=bool(args.ptq_layer_report_histogram),
            hist_bins=int(args.ptq_layer_report_hist_bins),
            hist_max_elems=int(args.ptq_layer_report_hist_max_elems),
        )
        try:
            with open(ptq_layer_report_path_resolved, encoding="utf-8") as rf:
                ptq_layer_report_data = json.load(rf)
        except Exception as exc:  # pragma: no cover
            logging.warning("读取 ptq layer report 嵌入 meta 失败: %s", exc)
            ptq_layer_report_data = {
                "error": str(exc),
                "path": str(ptq_layer_report_path_resolved),
            }
        _p("ptq_report", "分层 PTQ 报告已写入并完成读取")

    calib_collectors: list[Any] | None = None
    if args.calib_save_path is not None:
        calib_ok = args.inference_mode == "pytorch" or args.compare_mode or args.ptq_compare
        if not calib_ok:
            logging.warning(
                "已忽略 --calib-save-path：Pi0.5 calib 仅支持 pytorch / compare_mode / ptq_compare（当前为 %s）。",
                args.inference_mode,
            )
        else:
            try:
                _p("calib", "启动 Pi0.5 calib 收集器 …")
                calib_collectors = start_pi05_calib_collectors(
                    policy,
                    Path(args.calib_save_path),
                    max_samples=int(getattr(args, "calib_max_samples", 0)),
                    calib_item=str(getattr(args, "calib_item", "all")),
                )
                _p("calib", f"calib 收集器已启动 → {args.calib_save_path}")
            except Exception as exc:  # pragma: no cover
                logging.warning("启动 calib 收集失败，将继续评估但不保存 calib: %s", exc, exc_info=True)

    if args.compare_mode:
        backend = "pytorch+tensorrt"
    elif args.ptq_compare:
        backend = "pytorch+ptq"
    elif getattr(args, "ptq_trt_compare", False):
        backend = "pytorch_ptq+tensorrt"
    elif getattr(args, "trt_ort_compare", False):
        backend = "tensorrt+onnxrt"
    elif getattr(args, "trt_trt_compare", False):
        backend = "tensorrt+tensorrt"
    elif getattr(args, "ort_compare", False):
        backend = "pytorch+onnxrt"
    else:
        backend = args.inference_mode
    meta_payload: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": data_config.repo_id,
        "backend": backend,
        "compare_mode": bool(args.compare_mode),
        "vit_pt_trt_compare": bool(getattr(args, "vit_pt_trt_compare", False)),
        "ptq_compare": bool(args.ptq_compare),
        "ptq_trt_compare": bool(getattr(args, "ptq_trt_compare", False)),
        "ort_compare": bool(getattr(args, "ort_compare", False)),
        "trt_ort_compare": bool(getattr(args, "trt_ort_compare", False)),
        "trt_trt_compare": bool(getattr(args, "trt_trt_compare", False)),
        "action_horizon": int(action_horizon),
        "start_index": int(args.start_index),
        "end_index_exclusive": int(end),
        "send_wrist": bool(args.send_wrist),
        "jpeg_quality": int(args.jpeg_quality),
    }
    if args.calib_save_path is not None and calib_collectors is not None:
        meta_payload["calib_save_path"] = str(Path(args.calib_save_path).expanduser().resolve())

    if args.gpu_stats_interval_sec and args.gpu_stats_interval_sec > 0:
        meta_payload["gpu_stats_interval_sec"] = float(args.gpu_stats_interval_sec)
        meta_payload["gpu_device_index"] = int(effective_gpu_index(args))
    meta_payload["rel_err_denominator"] = "max_abs_gt_eps"
    meta_payload["rel_eps"] = float(args.rel_eps)
    meta_payload["flow_match_noise"] = str(args.noise)
    if args.noise == "fixed":
        meta_payload["flow_match_noise_seed"] = int(args.noise_seed)

    if args.ptq_compare:
        meta_payload["ptq_parts"] = list(args.ptq_parts)
        meta_payload["ptq_quant_cfg"] = str(Path(args.ptq_quant_cfg).expanduser().resolve())
        meta_payload["ptq_calib_dir"] = str(Path(args.ptq_calib_dir).expanduser().resolve())
        if args.ptq_layer_report_path is not None and ptq_layer_report_path_resolved is not None:
            meta_payload["ptq_layer_report_path"] = str(ptq_layer_report_path_resolved)
            if ptq_layer_report_data is not None:
                meta_payload["ptq_layer_report"] = ptq_layer_report_data
    if getattr(args, "ptq_trt_compare", False):
        meta_payload["ptq_parts"] = list(args.ptq_parts)
        meta_payload["ptq_quant_cfg"] = str(Path(args.ptq_quant_cfg).expanduser().resolve())
        meta_payload["ptq_calib_dir"] = str(Path(args.ptq_calib_dir).expanduser().resolve())

        # UI labels：pred1=PTQ(Pytorch), pred2=TRT(engine)
        meta_payload["pred1_name"] = "PTQ"
        meta_payload["pred2_name"] = "TRT"
        meta_payload["pair_name"] = "PTQ−TRT"
    elif args.ptq_compare:
        meta_payload["pred1_name"] = "PT"
        meta_payload["pred2_name"] = "PTQ"
        meta_payload["pair_name"] = "PT−PTQ"
    elif args.compare_mode:
        meta_payload["pred1_name"] = "PT"
        meta_payload["pred2_name"] = "TRT"
        meta_payload["pair_name"] = "PT−TRT"
    elif getattr(args, "ort_compare", False):
        meta_payload["pred1_name"] = "PT"
        meta_payload["pred2_name"] = "ORT"
        meta_payload["pair_name"] = "PT−ORT"
    elif getattr(args, "trt_ort_compare", False):
        meta_payload["pred1_name"] = "TRT"
        meta_payload["pred2_name"] = "ORT"
        meta_payload["pair_name"] = "TRT−ORT"
    elif getattr(args, "trt_trt_compare", False):
        meta_payload["pred1_name"] = "TRT_ref"
        meta_payload["pred2_name"] = "TRT_tgt"
        meta_payload["pair_name"] = "TRT_ref−TRT_tgt"

    if (
        args.inference_mode == "tensorrt"
        or args.compare_mode
        or getattr(args, "ptq_trt_compare", False)
        or getattr(args, "trt_ort_compare", False)
        or getattr(args, "trt_trt_compare", False)
    ):
        meta_payload["tensorrt"] = {
            "precision": args.precision,
            "engine_path": args.engine_path or "",
            "vit_engine": args.vit_engine or "",
            "llm_engine": args.llm_engine or "",
            "expert_engine": args.expert_engine or "",
            "denoise_engine": args.denoise_engine or "",
            "embed_prefix_engine": args.embed_prefix_engine or "",
        }
        if getattr(args, "trt_trt_compare", False):
            sec_nm = _trt_trt_second_engine_filenames(args)
            meta_payload["tensorrt"]["trt_trt_second_engine_path"] = str(
                getattr(args, "trt_trt_second_engine_path", "") or "",
            )
            meta_payload["tensorrt"]["trt_trt_second_vit_engine"] = sec_nm["vit_engine"] or ""
            meta_payload["tensorrt"]["trt_trt_second_llm_engine"] = sec_nm["llm_engine"] or ""
            meta_payload["tensorrt"]["trt_trt_second_expert_engine"] = sec_nm["expert_engine"] or ""
            meta_payload["tensorrt"]["trt_trt_second_denoise_engine"] = sec_nm["denoise_engine"] or ""
            meta_payload["tensorrt"]["trt_trt_second_embed_prefix_engine"] = sec_nm["embed_prefix_engine"] or ""

    if (
        args.inference_mode == "onnxrt"
        or getattr(args, "ort_compare", False)
        or getattr(args, "trt_ort_compare", False)
    ):
        meta_payload["onnxrt"] = {
            "engine_path": getattr(args, "ort_engine_path", "") or "",
            "vit_engine": getattr(args, "ort_vit_engine", "") or "",
            "llm_engine": getattr(args, "ort_llm_engine", "") or "",
            "expert_engine": getattr(args, "ort_expert_engine", "") or "",
            "denoise_engine": getattr(args, "ort_denoise_engine", "") or "",
            "embed_prefix_engine": getattr(args, "ort_embed_prefix_engine", "") or "",
        }

    trt_ort_polygraphy_report: dict[str, Any] | None = None
    if getattr(args, "trt_ort_compare", False) and getattr(args, "trt_ort_polygraphy_compare", False):
        from .trt_ort_polygraphy_compare import build_trt_ort_polygraphy_report

        _p("trt_ort_polygraphy", "Polygraphy：子图 TRT vs ORT 对比（可能较慢）…")
        trt_ort_polygraphy_report = build_trt_ort_polygraphy_report(args)
        ok_pg = bool(trt_ort_polygraphy_report.get("ok")) if isinstance(trt_ort_polygraphy_report, dict) else False
        _p(
            "trt_ort_polygraphy",
            "Polygraphy：对比完成（"
            + ("全部子图通过" if ok_pg else "存在失败、缺依赖或未配置子图；见 meta.trt_ort_polygraphy")
            + "）",
        )
    if trt_ort_polygraphy_report is not None:
        meta_payload["trt_ort_polygraphy"] = trt_ort_polygraphy_report

    _p("ready", "组装 meta、运行态统计器 …")
    meta_msg = event_to_json(meta_payload)
    _p("ready", "加载阶段完成，即将推送 meta 与 step 流")

    out: dict[str, Any] = {
        "meta_msg": meta_msg,
        "dataset": dataset,
        "repack_fn": repack_fn,
        "policy": policy,
        "policy_trt": policy_trt,
        "policy_ptq": policy_ptq,
        "n": n,
        "end": end,
        "start_index": int(args.start_index),
        "action_horizon": int(action_horizon),
        "action_dim": int(action_dim),
        "ep_per_frame": ep_per_frame,
        "run_id": run_id,
        "args": args,
        "calib_collectors": calib_collectors,
        "running_err_stats": RunningErrorStats(),
        "running_per_dim_mse_pct": RunningPerDimMsePctStats(),
        "running_per_dim_rel_p99": RunningPerDimRelP99Stats(),
        "running_vit_pt_trt": RunningVitCompareStats(),
        "trt_ort_compare": bool(getattr(args, "trt_ort_compare", False)),
        "trt_trt_compare": bool(getattr(args, "trt_trt_compare", False)),
    }
    _second_path_trt_style_stats = (
        bool(args.compare_mode)
        or bool(getattr(args, "ort_compare", False))
        or bool(getattr(args, "trt_ort_compare", False))
        or bool(getattr(args, "trt_trt_compare", False))
        or bool(getattr(args, "ptq_trt_compare", False))
    )
    if _second_path_trt_style_stats:
        out["running_per_dim_mse_pct_trt"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_trt"] = RunningPerDimRelP99Stats()
        out["running_pt_trt_mse_per_dim"] = RunningPerDimPairMseStats()
    if args.ptq_compare:
        out["running_per_dim_mse_pct_ptq"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_ptq"] = RunningPerDimRelP99Stats()
        out["running_pt_ptq_mse_per_dim"] = RunningPerDimPairMseStats()
    return out
