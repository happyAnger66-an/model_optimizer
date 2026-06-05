"""Synthetic kernel benchmark for the FlashRT Pi0.5 decoder.

This module builds the same pointer-level workload shape used by
``pipeline.decoder_forward`` and times every FVK call with CUDA events.

The default benchmark is a trace benchmark: it executes the full 10-step decoder
loop in the real kernel order, so every kernel receives inputs produced by its
actual upstream kernel.  Tensor values are synthetic, but shapes, dtypes, pointer
offsets, and buffer reuse match the production FlashRT decoder path.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .driver import DecoderBuffers, build_decoder_dims
from .kernels import load_kernels
from .precompute import build_dec_rope

_FP16 = torch.float16
_FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class KernelWorkload:
    """Approximate work metadata for one timed kernel launch."""

    flops: float = 0.0
    bytes: float = 0.0
    tokens: float = 0.0


@dataclass
class KernelSample:
    """One CUDA-event timing sample."""

    name: str
    elapsed_ms: float
    workload: KernelWorkload
    step: int | None = None
    layer: int | None = None


@dataclass
class KernelSummary:
    """Aggregated statistics for one kernel site."""

    name: str
    calls: int
    mean_us: float
    p50_us: float
    p90_us: float
    p99_us: float
    total_ms: float
    tflops: float | None
    gbps: float | None
    tokens_per_s: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p90_us": self.p90_us,
            "p99_us": self.p99_us,
            "total_ms": self.total_ms,
            "tflops": self.tflops,
            "gbps": self.gbps,
            "tokens_per_s": self.tokens_per_s,
        }


class CudaKernelBenchRecorder:
    """Records FVK call latency using CUDA events."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.samples: list[KernelSample] = []

    def time(
        self,
        name: str,
        workload: KernelWorkload,
        fn: Callable[[], None],
        *,
        step: int | None = None,
        layer: int | None = None,
    ) -> None:
        if not self.enabled:
            fn()
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        self.samples.append(
            KernelSample(
                name=name,
                elapsed_ms=float(start.elapsed_time(end)),
                workload=workload,
                step=step,
                layer=layer,
            )
        )

    def clear(self) -> None:
        self.samples.clear()

    def summaries(self) -> list[KernelSummary]:
        grouped: dict[str, list[KernelSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.name].append(sample)

        out: list[KernelSummary] = []
        for name, samples in grouped.items():
            ms = [s.elapsed_ms for s in samples]
            total_ms = float(sum(ms))
            total_s = total_ms / 1000.0
            total_flops = sum(s.workload.flops for s in samples)
            total_bytes = sum(s.workload.bytes for s in samples)
            total_tokens = sum(s.workload.tokens for s in samples)
            out.append(
                KernelSummary(
                    name=name,
                    calls=len(samples),
                    mean_us=statistics.fmean(ms) * 1000.0,
                    p50_us=_percentile(ms, 50) * 1000.0,
                    p90_us=_percentile(ms, 90) * 1000.0,
                    p99_us=_percentile(ms, 99) * 1000.0,
                    total_ms=total_ms,
                    tflops=(total_flops / total_s / 1e12) if total_flops and total_s else None,
                    gbps=(total_bytes / total_s / 1e9) if total_bytes and total_s else None,
                    tokens_per_s=(total_tokens / total_s) if total_tokens and total_s else None,
                )
            )
        return sorted(out, key=lambda x: x.total_ms, reverse=True)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "samples": [
                {
                    "name": s.name,
                    "elapsed_ms": s.elapsed_ms,
                    "step": s.step,
                    "layer": s.layer,
                    "workload": {
                        "flops": s.workload.flops,
                        "bytes": s.workload.bytes,
                        "tokens": s.workload.tokens,
                    },
                }
                for s in self.samples
            ],
            "summary": [s.as_dict() for s in self.summaries()],
        }


@dataclass
class SyntheticDecoderBenchmarkState:
    """Holds synthetic tensors and raw pointer dictionaries for benchmarking."""

    dims: dict[str, int]
    bufs: DecoderBuffers
    weights: dict[str, int]
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    initial_noise: torch.Tensor | None = None
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None

    def reset_inputs(self) -> None:
        """Restore mutable entry buffers before one full-loop replay."""
        if self.initial_noise is not None:
            self.bufs.noise.copy_(self.initial_noise)
        if self.prefix_k is not None:
            self.bufs.Kc[:, : self.dims["enc_seq"], :].copy_(self.prefix_k)
        if self.prefix_v is not None:
            self.bufs.Vc[:, : self.dims["enc_seq"], :].copy_(self.prefix_v)


def create_synthetic_decoder_state(
    *,
    S: int = 10,
    D: int = 1024,
    H: int = 4096,
    layers: int = 18,
    steps: int = 10,
    enc_seq: int = 818,
    NH: int = 8,
    HD: int = 256,
    device: str = "cuda",
    seed: int = 1234,
    force_even_enc_seq: bool = True,
) -> SyntheticDecoderBenchmarkState:
    """Create synthetic tensors with the same layout as ``decoder_forward``.

    Values are random but intentionally small to avoid numerical overflow in the
    softmax and normalization paths.  Shapes and pointer arithmetic mirror the
    production path exactly.
    """
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA is required for FlashRT decoder kernel benchmark")
    if force_even_enc_seq and enc_seq % 2:
        enc_seq += 1
    if HD % 2:
        raise ValueError(f"HD must be even for RoPE, got {HD}")

    torch.manual_seed(int(seed))
    dims = build_decoder_dims(Sa=S, Da=D, Ha=H, layers=layers, enc_seq=enc_seq, NH=NH, HD=HD, steps=steps)
    bufs = DecoderBuffers(Sa=S, Da=D, Ha=H, layers=layers, enc_seq=enc_seq, HD=HD, NH=NH, device=device)

    tensors: dict[str, torch.Tensor] = {}
    tensors["ain_w"] = _rand_fp16((32, D), device)
    tensors["ain_b"] = _rand_fp16((D,), device)
    tensors["sa"] = _rand_fp16((steps * layers * S, 3 * D), device)
    tensors["sf"] = _rand_fp16((steps * layers * S, 3 * D), device)
    tensors["fs"] = _rand_fp16((steps * S, 3 * D), device)
    tensors["rope"] = build_dec_rope(enc_seq, S, device=device, head_dim=HD, max_pos=max(1200, enc_seq + S))

    qkv, s_qkv = _rand_fp8((layers, D, 2560), device)
    ow, s_o = _rand_fp8((layers, NH * HD, D), device)
    gu, s_gu = _rand_fp8((layers, D, 2 * H), device)
    dw, s_dw = _rand_fp8((layers, H, D), device)
    tensors["qw"] = qkv.reshape(-1).contiguous()
    tensors["ow"] = ow.reshape(-1).contiguous()
    tensors["gw"] = gu.reshape(-1).contiguous()
    tensors["dw"] = dw.reshape(-1).contiguous()

    w_scales = torch.empty(layers * 4, dtype=torch.float32, device=device)
    for layer in range(layers):
        w_scales[layer * 4 + 0] = s_qkv[layer]
        w_scales[layer * 4 + 1] = s_o[layer]
        w_scales[layer * 4 + 2] = s_gu[layer]
        w_scales[layer * 4 + 3] = s_dw[layer]
    tensors["w_scales"] = w_scales
    tensors["act_scales"] = torch.full((layers * 4,), 1.0, dtype=torch.float32, device=device)

    tensors["aow"] = _rand_fp16((D, 32), device)
    tensors["aob"] = _rand_fp16((32,), device)

    initial_noise = _rand_fp16((S, 32), device)
    prefix_k = _rand_fp16((layers, enc_seq, HD), device)
    prefix_v = _rand_fp16((layers, enc_seq, HD), device)

    weights = {
        "ain_w": tensors["ain_w"].reshape(-1).data_ptr(),
        "ain_b": tensors["ain_b"].reshape(-1).data_ptr(),
        "sa": tensors["sa"].reshape(-1).data_ptr(),
        "qw": tensors["qw"].reshape(-1).data_ptr(),
        "Kc": bufs.Kc.reshape(-1).data_ptr(),
        "Vc": bufs.Vc.reshape(-1).data_ptr(),
        "ow": tensors["ow"].reshape(-1).data_ptr(),
        "sf": tensors["sf"].reshape(-1).data_ptr(),
        "gw": tensors["gw"].reshape(-1).data_ptr(),
        "dw": tensors["dw"].reshape(-1).data_ptr(),
        "aow": tensors["aow"].reshape(-1).data_ptr(),
        "aob": tensors["aob"].reshape(-1).data_ptr(),
        "fs": tensors["fs"].reshape(-1).data_ptr(),
        "rope": tensors["rope"].reshape(-1).data_ptr(),
        "w_scales": tensors["w_scales"].reshape(-1).data_ptr(),
        "act_scales": tensors["act_scales"].reshape(-1).data_ptr(),
    }
    state = SyntheticDecoderBenchmarkState(
        dims=dims,
        bufs=bufs,
        weights=weights,
        tensors=tensors,
        initial_noise=initial_noise,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
    )
    state.reset_inputs()
    return state


def decoder_forward_trace(
    ctx: Any,
    fvk: Any,
    bufs: dict[str, int],
    weights: dict[str, int],
    dims: dict[str, int],
    *,
    recorder: CudaKernelBenchRecorder | None,
    stream: int = 0,
) -> None:
    """Run the FP8 decoder path with per-FVK-call CUDA event timings."""
    S = dims["S"]
    D = dims["D"]
    H = dims["H"]
    NH = dims["NH"]
    HD = dims["HD"]
    steps = dims["steps"]
    layers = dims["layers"]
    enc_seq = dims["enc_seq"]
    total_keys = dims["total_keys"]
    D3 = 3 * D
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    noise = bufs["noise"]
    x = bufs["x"]
    xn = bufs["xn"]
    gate = bufs["gate"]
    qkv = bufs["qkv"]
    logits = bufs["logits"]
    attn_out = bufs["attn_out"]
    fg = bufs["fg"]
    xn_fp8 = bufs["xn_fp8"]
    hid_fp8 = bufs["hid_fp8"]
    ctx_fp8 = bufs["ctx_fp8"]

    ain_w = weights["ain_w"]
    ain_b = weights["ain_b"]
    sa = weights["sa"]
    qw = weights["qw"]
    Kc = weights["Kc"]
    Vc = weights["Vc"]
    ow = weights["ow"]
    sf = weights["sf"]
    gw = weights["gw"]
    dw = weights["dw"]
    aow = weights["aow"]
    aob = weights["aob"]
    fs = weights["fs"]
    rope = weights["rope"]
    w_scales = weights["w_scales"]
    act_scales = weights["act_scales"]

    rec = recorder or CudaKernelBenchRecorder(enabled=False)

    for step in range(steps):
        rec.time(
            "gmm_fp16.action_in",
            _gemm_work(S, D, 32, dtype_bytes=2),
            lambda: fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream),
            step=step,
        )
        rec.time(
            "add_bias_fp16.action_in",
            KernelWorkload(bytes=(S * D * 2 * 2) + D * 2, tokens=S),
            lambda: fvk.add_bias_fp16(x, ain_b, S, D, stream),
            step=step,
        )

        for layer in range(layers):
            si = (step * layers + layer) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            act_scale_qkv = act_scales + (layer * 4 + 0) * 4
            rec.time(
                "fused_adarms_fp8_static_fp16.C1",
                KernelWorkload(bytes=S * D * (2 + 3 * 2 + 1 + 2), tokens=S),
                lambda: fvk.fused_adarms_fp8_static_fp16(x, sa_ptr, xn_fp8, gate, S, D, act_scale_qkv, stream),
                step=step,
                layer=layer,
            )

            w_scale_qkv = w_scales + (layer * 4 + 0) * 4
            qw_ptr = qw + layer * D * 2560
            rec.time(
                "fp8_gemm_descale_fp16.C2.qkv",
                _gemm_work(S, 2560, D, dtype_bytes=1, out_bytes=2),
                lambda: fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D, act_scale_qkv, w_scale_qkv, stream),
                step=step,
                layer=layer,
            )

            kv_offset = layer * total_keys * HD + enc_seq * HD
            rec.time(
                "qkv_split_rope_kvcache_fp16.C2b",
                _qkv_split_work(S, Q_dim, K_dim, HD),
                lambda: fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc, S, Q_dim, K_dim, HD, 2560, kv_offset, HD, stream),
                step=step,
                layer=layer,
            )

            K_ptr = Kc + layer * total_keys * HD * 2
            V_ptr = Vc + layer * total_keys * HD * 2
            rec.time(
                "attention_qkv_fp16.C3",
                _attention_work(S, total_keys, NH, HD),
                lambda: fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr, logits, attn_out, S, total_keys, NH, HD, attn_scale, stream),
                step=step,
                layer=layer,
            )

            act_scale_o = act_scales + (layer * 4 + 1) * 4
            rec.time(
                "quantize_fp8_static_fp16.C4.attn_out",
                KernelWorkload(bytes=S * NH * HD * (2 + 1), tokens=S),
                lambda: fvk.quantize_fp8_static_fp16(attn_out, ctx_fp8, act_scale_o, S * NH * HD, stream),
                step=step,
                layer=layer,
            )

            w_scale_o = w_scales + (layer * 4 + 1) * 4
            ow_ptr = ow + layer * NH * HD * D
            rec.time(
                "fp8_gemm_descale_fp16.C4.o",
                _gemm_work(S, D, NH * HD, dtype_bytes=1, out_bytes=2),
                lambda: fvk.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD, act_scale_o, w_scale_o, stream),
                step=step,
                layer=layer,
            )

            act_scale_gu = act_scales + (layer * 4 + 2) * 4
            rec.time(
                "gate_res_adarms_fp8_static_fp16.C4C5",
                KernelWorkload(bytes=S * D * (2 * 4 + 3 * 2 + 1), tokens=S),
                lambda: fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sf_ptr, xn_fp8, gate, S, D, act_scale_gu, stream),
                step=step,
                layer=layer,
            )

            w_scale_gu = w_scales + (layer * 4 + 2) * 4
            gw_ptr = gw + layer * D * H * 2
            rec.time(
                "fp8_gemm_descale_fp16.C5.gate_up",
                _gemm_work(S, H * 2, D, dtype_bytes=1, out_bytes=2),
                lambda: fvk.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D, act_scale_gu, w_scale_gu, stream),
                step=step,
                layer=layer,
            )

            act_scale_down = act_scales + (layer * 4 + 3) * 4
            rec.time(
                "gate_geglu_merged_fp8_fp16.C6",
                KernelWorkload(bytes=S * H * (2 * 2 + 1), tokens=S),
                lambda: fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, act_scale_down, stream),
                step=step,
                layer=layer,
            )

            w_scale_down = w_scales + (layer * 4 + 3) * 4
            dw_ptr = dw + layer * H * D
            rec.time(
                "fp8_gemm_descale_fp16.C6.down",
                _gemm_work(S, D, H, dtype_bytes=1, out_bytes=2),
                lambda: fvk.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S, D, H, act_scale_down, w_scale_down, stream),
                step=step,
                layer=layer,
            )

            if layer < layers - 1:
                si_next = (step * layers + layer + 1) * S * D3
                sa_next_ptr = sa + si_next * 2
                act_scale_next = act_scales + ((layer + 1) * 4 + 0) * 4
                rec.time(
                    "gate_res_adarms_fp8_static_fp16.C7.next",
                    KernelWorkload(bytes=S * D * (2 * 4 + 3 * 2 + 1), tokens=S),
                    lambda: fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sa_next_ptr, xn_fp8, gate, S, D, act_scale_next, stream),
                    step=step,
                    layer=layer,
                )
            else:
                rec.time(
                    "gate_res_fp16.C7.last",
                    KernelWorkload(bytes=S * D * (2 * 3), tokens=S),
                    lambda: fvk.gate_res_fp16(fg, gate, x, S * D, stream),
                    step=step,
                    layer=layer,
                )

        fi = step * S * D3
        fs_ptr = fs + fi * 2
        rec.time(
            "adarms_fp16.final",
            KernelWorkload(bytes=S * D * (2 + 3 * 2 + 2 + 2), tokens=S),
            lambda: fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream),
            step=step,
        )
        rec.time(
            "gmm_fp16.action_out",
            _gemm_work(S, 32, D, dtype_bytes=2),
            lambda: fvk.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream),
            step=step,
        )
        rec.time(
            "add_bias_fp16.action_out",
            KernelWorkload(bytes=(S * 32 * 2 * 2) + 32 * 2, tokens=S),
            lambda: fvk.add_bias_fp16(noise, aob, S, 32, stream),
            step=step,
        )


def run_synthetic_trace_benchmark(
    *,
    build_dir: str | None = None,
    fmha_so: str | None = None,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cuda",
    stream: int = 0,
    **shape_kwargs: Any,
) -> tuple[SyntheticDecoderBenchmarkState, CudaKernelBenchRecorder]:
    """Build synthetic state and run the trace benchmark."""
    if stream != 0:
        raise ValueError("Only stream=0 is currently supported by the Python CUDA-event recorder")
    fvk = load_kernels(build_dir, fmha_so=fmha_so)
    ctx = fvk.FvkContext()
    state = create_synthetic_decoder_state(device=device, **shape_kwargs)
    bufs = state.bufs.as_ptr_dict()

    warmup_recorder = CudaKernelBenchRecorder(enabled=False)
    for _ in range(max(0, int(warmup))):
        state.reset_inputs()
        decoder_forward_trace(ctx, fvk, bufs, state.weights, state.dims, recorder=warmup_recorder, stream=stream)
    torch.cuda.synchronize()

    recorder = CudaKernelBenchRecorder(enabled=True)
    for _ in range(max(1, int(iters))):
        state.reset_inputs()
        decoder_forward_trace(ctx, fvk, bufs, state.weights, state.dims, recorder=recorder, stream=stream)
    torch.cuda.synchronize()
    return state, recorder


def format_summary_table(summaries: list[KernelSummary]) -> str:
    """Format benchmark summaries as a plain text table."""
    header = (
        f"{'kernel':<46} {'calls':>7} {'mean_us':>10} {'p50_us':>10} "
        f"{'p90_us':>10} {'total_ms':>10} {'TFLOP/s':>10} {'GB/s':>10}"
    )
    lines = [header, "-" * len(header)]
    for item in summaries:
        tflops = "-" if item.tflops is None else f"{item.tflops:.2f}"
        gbps = "-" if item.gbps is None else f"{item.gbps:.1f}"
        lines.append(
            f"{item.name:<46} {item.calls:>7d} {item.mean_us:>10.2f} "
            f"{item.p50_us:>10.2f} {item.p90_us:>10.2f} {item.total_ms:>10.2f} "
            f"{tflops:>10} {gbps:>10}"
        )
    return "\n".join(lines)


def write_benchmark_json(path: str | Path, *, state: SyntheticDecoderBenchmarkState, recorder: CudaKernelBenchRecorder) -> None:
    out = {
        "dims": state.dims,
        **recorder.to_json_dict(),
    }
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


def _rand_fp16(shape: tuple[int, ...], device: str, *, scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(_FP16).contiguous()


def _rand_fp8(shape: tuple[int, ...], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(shape, device=device, dtype=torch.float32) * 0.02
    flat = x.reshape(shape[0], -1)
    scale = torch.clamp(flat.abs().amax(dim=1) / 448.0, min=1e-12).to(torch.float32)
    q = (flat / scale[:, None]).clamp(-448, 448).reshape(shape).to(_FP8).contiguous()
    return q, scale


def _gemm_work(M: int, N: int, K: int, *, dtype_bytes: int, out_bytes: int | None = None) -> KernelWorkload:
    out_b = dtype_bytes if out_bytes is None else out_bytes
    return KernelWorkload(
        flops=2.0 * M * N * K,
        bytes=(M * K + K * N) * dtype_bytes + M * N * out_b,
        tokens=M,
    )


def _attention_work(S: int, S_kv: int, NH: int, HD: int) -> KernelWorkload:
    qk_flops = 2.0 * S * NH * S_kv * HD
    pv_flops = 2.0 * S * NH * S_kv * HD
    logits_bytes = S * NH * S_kv * 2
    qkv_bytes = (S * NH * HD + 2 * S_kv * HD + S * NH * HD) * 2
    return KernelWorkload(flops=qk_flops + pv_flops, bytes=qkv_bytes + logits_bytes * 2, tokens=S)


def _qkv_split_work(S: int, Q_dim: int, K_dim: int, HD: int) -> KernelWorkload:
    v_dim = K_dim
    qkv_read = S * (Q_dim + K_dim + v_dim) * 2
    rope_read = S * HD * 2
    writes = S * (Q_dim + K_dim + v_dim) * 2
    return KernelWorkload(bytes=qkv_read + rope_read + writes, tokens=S)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)
