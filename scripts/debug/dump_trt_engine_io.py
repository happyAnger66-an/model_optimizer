import argparse
from dataclasses import dataclass
from typing import Any

import torch

from model_optimizer.infer.tensorrt.trt_torch import Engine


@dataclass
class Stats:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    min: float
    max: float
    mean: float
    abs_sum: float


def _tensor_stats(name: str, x: torch.Tensor) -> Stats:
    with torch.no_grad():
        xf = x.detach()
        # 对 int/bool 也能统计，但用 float 转一下
        if xf.numel() == 0:
            mn = mx = mu = s = 0.0
        else:
            xff = xf.float()
            mn = float(xff.min().item())
            mx = float(xff.max().item())
            mu = float(xff.mean().item())
            s = float(xff.abs().sum().item())
        return Stats(
            name=name,
            shape=tuple(int(d) for d in xf.shape),
            dtype=str(xf.dtype),
            device=str(xf.device),
            min=mn,
            max=mx,
            mean=mu,
            abs_sum=s,
        )


def _default_dim(dim_name: str) -> int:
    # 尽量给小一点的默认，避免显存爆
    if dim_name in ("batch", "batch_size", "B"):
        return 1
    if dim_name in ("seq", "seq_len", "S"):
        return 8
    if dim_name in ("past_len", "P"):
        return 2
    return 1


def _fill_dynamic(shape: tuple[int, ...], dynamic: dict[str, int]) -> tuple[int, ...]:
    out = []
    for d in shape:
        if int(d) < 0:
            # TensorRT 动态维通常是 -1：按约定映射
            out.append(dynamic.get("seq_len", _default_dim("seq_len")))
        else:
            out.append(int(d))
    return tuple(out)


def _make_attention_mask(
    bsz: int,
    seq: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    mask_value: float,
) -> torch.Tensor:
    # 模式说明：
    # - none: 全 0（不屏蔽）
    # - causal_float: 上三角为 mask_value（默认 -1e4），下三角为 0；用于“加到 logits 上”的浮点 mask
    if mode == "none":
        return torch.zeros((bsz, 1, seq, seq), device=device, dtype=dtype)
    if mode != "causal_float":
        raise ValueError(f"Unknown attention_mask mode: {mode}")
    v = mask_value
    mask = torch.triu(torch.full((seq, seq), v, device=device, dtype=dtype), diagonal=1)
    return mask.view(1, 1, seq, seq).expand(bsz, 1, seq, seq).contiguous()


def _make_inputs_from_signature(
    engine: Engine,
    *,
    seq_len: int,
    batch_size: int,
    past_len: int,
    device: str,
    attention_mask_mode: str,
    attention_mask_value: float,
) -> dict[str, Any]:
    dev = torch.device(device)
    inputs: dict[str, Any] = {}
    for name, trt_shape, dtype in engine.in_meta:
        # 把动态维替换成可跑的实际数值
        shape = _fill_dynamic(tuple(trt_shape), {"seq_len": seq_len})

        if name.startswith("past_key_values."):
            # 约定：[B,2,Hkv,past_len,D]
            # 若 engine 里 past_len 是动态维，这里用用户传入
            shape = tuple(past_len if (i == 3 and int(trt_shape[i]) < 0) else shape[i] for i in range(len(shape)))
            x = torch.zeros(shape, device=dev, dtype=dtype)
            inputs[name] = x
            continue

        if name == "inputs_embeds":
            x = torch.randn(shape, device=dev, dtype=dtype) * 0.02
            inputs[name] = x
            continue

        if name == "attention_mask":
            bsz = batch_size if (len(shape) > 0 and shape[0] in (-1, batch_size)) else shape[0]
            seq = seq_len
            inputs[name] = _make_attention_mask(
                bsz,
                seq,
                device=dev,
                dtype=dtype,
                mode=attention_mask_mode,
                mask_value=attention_mask_value,
            )
            continue

        if name == "position_ids":
            # [B,S] int64
            pid = torch.arange(seq_len, device=dev, dtype=torch.long).view(1, -1).expand(batch_size, -1).contiguous()
            if dtype != torch.long:
                pid = pid.to(dtype)
            inputs[name] = pid
            continue

        # 兜底：全 0
        if dtype == torch.int32 or dtype == torch.int64:
            inputs[name] = torch.zeros(shape, device=dev, dtype=dtype)
        else:
            inputs[name] = torch.zeros(shape, device=dev, dtype=dtype)

    return inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to .engine")
    ap.add_argument("--device", default="cuda", help="cuda/cuda:0/cpu")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--past_len", type=int, default=2)
    ap.add_argument(
        "--attention_mask_mode",
        choices=["none", "causal_float"],
        default="causal_float",
        help="How to construct attention_mask input (if present).",
    )
    ap.add_argument(
        "--attention_mask_value",
        type=float,
        default=-1e4,
        help="Value used for masked positions when attention_mask_mode=causal_float.",
    )
    ap.add_argument("--print_values", action="store_true", help="Print first few values")
    args = ap.parse_args()

    eng = Engine(args.engine, perf=False)
    inputs = _make_inputs_from_signature(
        eng,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        past_len=args.past_len,
        device=args.device,
        attention_mask_mode=args.attention_mask_mode,
        attention_mask_value=args.attention_mask_value,
    )

    with torch.inference_mode():
        outputs = eng(**inputs)

    if not isinstance(outputs, dict):
        raise TypeError(f"Engine returned {type(outputs)}, expected dict[str,Tensor]")

    print("\n=== Inputs ===")
    for name, x in inputs.items():
        st = _tensor_stats(name, x)
        print(
            f"{st.name}: shape={st.shape} dtype={st.dtype} device={st.device} "
            f"min={st.min:.4g} max={st.max:.4g} mean={st.mean:.4g} abs_sum={st.abs_sum:.4g}"
        )

    print("\n=== Outputs ===")
    for name, x in outputs.items():
        st = _tensor_stats(name, x)
        print(
            f"{st.name}: shape={st.shape} dtype={st.dtype} device={st.device} "
            f"min={st.min:.4g} max={st.max:.4g} mean={st.mean:.4g} abs_sum={st.abs_sum:.4g}"
        )
        if args.print_values:
            flat = x.reshape(-1)
            n = min(int(flat.numel()), 16)
            vals = flat[:n].detach().cpu()
            print(f"  head({n}) = {vals.tolist()}")


if __name__ == "__main__":
    main()

