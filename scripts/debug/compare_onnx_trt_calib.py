#!/usr/bin/env python3
"""使用 ``torch.save`` 的校准/采集数据，分别跑 ONNX 与 TensorRT engine，并对比输出差异。

采集数据格式与 ``src/model_optimizer/calibrate/collector/pi05.py`` 中各 Collector 一致：

- **llm**（``Pi05LLMCalibCollector``）：``list[dict]``，每项含
  ``inputs_embeds``, ``attention_mask``, ``position_ids``（CPU tensor）。
- **expert**（``Pi05ExpertCalibCollector``）：``list[dict]``，每项含
  ``attention_mask``, ``position_ids``, ``inputs_embeds``, ``adarms_cond``（可选）,
  ``input_keys``, ``input_values``（由 ``past_key_values`` 展开；与 ONNX 中
  ``past_keys`` / ``past_values`` 对应）。
- **vit**（``Pi05VitCalibCollector``）：``list[Tensor]``，每项为 ``pixel_values``。

示例::

    PYTHONPATH=src python scripts/debug/compare_onnx_trt_calib.py \\
      --calib_pt /path/pi05_llm_calib_datas.pt \\
      --onnx /path/llm.onnx \\
      --engine /path/llm.engine \\
      --graph llm

依赖：``onnxruntime``、TensorRT Python、与 engine 匹配的插件（expert 等需 ``--plugin``）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import torch

from model_optimizer.evaluate.compare.utils import compare_predictions
from model_optimizer.infer.tensorrt.trt_torch import Engine


def _warn(s: str) -> None:
    try:
        from termcolor import colored

        print(colored(s, "yellow"), file=sys.stderr)
    except Exception:
        print(s, file=sys.stderr)


def _load_calib_file(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_to_ort_numpy(t: torch.Tensor) -> np.ndarray:
    """ORT 常用 float32；bf16/fp16 先转 float32 numpy。"""
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.detach().float().cpu().numpy()
    if t.dtype in (torch.float32, torch.float64):
        return np.asarray(t.detach().cpu().numpy(), dtype=np.float32)
    if t.dtype in (torch.int32, torch.int64):
        return t.detach().cpu().numpy()
    return t.detach().cpu().numpy()


def _tensor_to_cuda(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.contiguous().to(device=device)


def _normalize_sample(graph: str, item: Any) -> dict[str, torch.Tensor]:
    if graph == "vit":
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"vit 模式期望 list 中每项为 Tensor，得到 {type(item)}"
            )
        return {"pixel_values": item}

    if not isinstance(item, dict):
        raise TypeError(
            f"{graph} 模式期望 list 中每项为 dict，得到 {type(item)}"
        )

    if graph == "llm":
        need = {"inputs_embeds", "attention_mask", "position_ids"}
        missing = need - set(item.keys())
        if missing:
            raise KeyError(f"llm 样本缺少键: {missing}")
        return {k: item[k] for k in need}

    if graph == "expert":
        keys = set(item.keys())
        need = {"attention_mask", "position_ids", "inputs_embeds", "input_keys", "input_values"}
        missing = need - keys
        if missing:
            raise KeyError(f"expert 样本缺少键: {missing}")
        out = {
            "attention_mask": item["attention_mask"],
            "position_ids": item["position_ids"],
            "inputs_embeds": item["inputs_embeds"],
            "input_keys": item["input_keys"],
            "input_values": item["input_values"],
        }
        if "adarms_cond" in item and item["adarms_cond"] is not None:
            out["adarms_cond"] = item["adarms_cond"]
        return out

    raise ValueError(f"未知 graph: {graph}")


def _expert_fill_adarms_if_missing(sample: dict[str, torch.Tensor]) -> None:
    if sample.get("adarms_cond") is not None:
        return
    ie = sample["inputs_embeds"]
    dim = int(ie.shape[-1])
    b = int(ie.shape[0])
    sample["adarms_cond"] = torch.zeros(
        b, dim, dtype=ie.dtype, device=ie.device
    )


def _map_expert_to_onnx_trt_names(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collector 使用 input_keys/input_values；ONNX/TRT 多为 past_keys/past_values。"""
    out = {}
    for k, v in sample.items():
        if k == "input_keys":
            out["past_keys"] = v
        elif k == "input_values":
            out["past_values"] = v
        else:
            out[k] = v
    return out


def _build_ort_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "需要安装 onnxruntime: pip install onnxruntime 或 onnxruntime-gpu"
        ) from e

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def _run_onnx(
    sess,
    feeds_torch: dict[str, torch.Tensor],
    input_names: list[str],
) -> dict[str, np.ndarray]:
    feeds = {n: _torch_to_ort_numpy(feeds_torch[n]) for n in input_names}
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    return dict(zip(out_names, outs))


def _run_trt(
    engine: Engine,
    feeds_cuda: dict[str, torch.Tensor],
    in_order: list[str],
) -> dict[str, torch.Tensor]:
    kwargs = {name: feeds_cuda[name] for name in in_order}
    out = engine(**kwargs)
    if not isinstance(out, dict):
        raise TypeError(f"TRT 期望返回 dict，得到 {type(out)}")
    return out


def _outputs_to_numpy_pred(
    onnx_out: dict[str, np.ndarray],
    trt_out: dict[str, torch.Tensor],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """对齐两边输出名；仅对比公共键。"""
    common = sorted(set(onnx_out.keys()) & set(trt_out.keys()))
    if not common:
        raise ValueError(
            f"ONNX 与 TRT 输出名无交集。ONNX={list(onnx_out.keys())} TRT={list(trt_out.keys())}"
        )
    pred_onnx = {k: onnx_out[k] for k in common}
    pred_trt = {k: trt_out[k].detach().float().cpu().numpy() for k in common}
    return pred_trt, pred_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX vs TensorRT：使用 pi05 hook 采集的 torch.save 数据对比输出"
    )
    parser.add_argument(
        "--calib_pt",
        required=True,
        help="torch.save 的采集文件（如 pi05_llm_calib_datas.pt）",
    )
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument("--engine", required=True, help="TensorRT .engine 路径")
    parser.add_argument(
        "--graph",
        choices=("llm", "expert", "vit"),
        required=True,
        help="与采集脚本一致：llm / expert / vit",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多跑前 N 条；默认全部",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="TRT 输入张量设备，如 cuda:0",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="TensorRT 自定义插件 .so，可多次指定",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.calib_pt):
        print(f"找不到 calib 文件: {args.calib_pt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.onnx):
        print(f"找不到 ONNX: {args.onnx}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.engine):
        print(f"找不到 engine: {args.engine}", file=sys.stderr)
        sys.exit(1)

    raw = _load_calib_file(args.calib_pt)
    if isinstance(raw, list):
        samples = raw
    else:
        samples = [raw]

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    dev = torch.device(args.device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        print("TRT 推理需要 CUDA 张量。", file=sys.stderr)
        sys.exit(1)

    sess = _build_ort_session(args.onnx)
    ort_in_names = [i.name for i in sess.get_inputs()]

    trt_eng = Engine(args.engine, perf=False, plugins=args.plugin)
    trt_in_names = [m[0] for m in trt_eng.in_meta]

    if ort_in_names != trt_in_names:
        _warn(
            "ONNX 与 TRT 输入名/顺序不一致：\n"
            f"  ONNX: {ort_in_names}\n"
            f"  TRT : {trt_in_names}\n"
            "将分别按两侧名称从样本取张量；请确认与导出一致。"
        )

    for idx, item in enumerate(samples):
        print(f"\n{'=' * 60}\n样本 {idx + 1} / {len(samples)}\n{'=' * 60}")
        sample = _normalize_sample(args.graph, item)
        if args.graph == "expert":
            _expert_fill_adarms_if_missing(sample)
            sample = _map_expert_to_onnx_trt_names(sample)

        feeds_ort: dict[str, torch.Tensor] = {}
        for name in ort_in_names:
            if name not in sample:
                raise KeyError(
                    f"样本缺少 ONNX 输入 {name!r}；当前键: {list(sample.keys())}"
                )
            t = sample[name]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            feeds_ort[name] = t

        feeds_trt: dict[str, torch.Tensor] = {}
        for name in trt_in_names:
            if name not in sample:
                raise KeyError(
                    f"样本缺少 TRT 输入 {name!r}；当前键: {list(sample.keys())}"
                )
            t = sample[name]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            feeds_trt[name] = _tensor_to_cuda(t, dev)

        onnx_out = _run_onnx(sess, feeds_ort, ort_in_names)
        trt_raw = _run_trt(trt_eng, feeds_trt, trt_in_names)

        pred_trt, pred_onnx = _outputs_to_numpy_pred(onnx_out, trt_raw)
        compare_predictions(
            pred_trt,
            pred_onnx,
            key1="TensorRT",
            key2="ONNX",
            filter_keys=None,
            return_metrics=False,
        )


if __name__ == "__main__":
    main()
