"""FP8 language embedding sidecar for π0.5 (Edge-LLM compatible layout)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Align with TensorRT-Edge-LLM embedding_quantization.py / C++ kFP8EmbeddingBlockSize.
FP8_EMBEDDING_BLOCK_SIZE = 128
FP8_E4M3_MAX = 448.0

LANG_EMBEDDING_SIDECAR_FILENAME = "lang_embedding.safetensors"
LANG_EMBEDDING_META_FILENAME = "lang_embedding.meta.json"
EMBEDDING_TENSOR_KEY = "embedding"
EMBEDDING_SCALE_TENSOR_KEY = "embedding_scale"
META_FORMAT = "fp8_e4m3"


@dataclass(frozen=True)
class LangEmbeddingMeta:
    vocab_size: int
    hidden_size: int
    block_size: int
    format: str = META_FORMAT

    @property
    def table_bytes(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def scales_bytes(self) -> int:
        num_groups = self.hidden_size // self.block_size
        return self.vocab_size * num_groups * 4

    @property
    def total_bytes(self) -> int:
        return self.table_bytes + self.scales_bytes

    def fp16_baseline_bytes(self) -> int:
        return self.vocab_size * self.hidden_size * 2

    def saved_bytes_vs_fp16(self) -> int:
        return self.fp16_baseline_bytes() - self.total_bytes


@dataclass
class Fp8LangEmbeddingTables:
    embedding: torch.Tensor
    scales: torch.Tensor
    meta: LangEmbeddingMeta
    device: torch.device

    def memory_stats(self) -> LangEmbeddingMeta:
        return self.meta


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor,
    block_size: int = FP8_EMBEDDING_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize embedding table to FP8 E4M3 with per-row per-block scales."""
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D"
        )

    vocab_size, hidden_size = embedding_weight.shape
    if hidden_size % block_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by block size {block_size}"
        )

    num_groups = hidden_size // block_size
    weight_fp32 = embedding_weight.float()
    weight_reshaped = weight_fp32.view(vocab_size, num_groups, block_size)
    amax = weight_reshaped.abs().amax(dim=-1).clamp(min=1e-4)
    scales = amax / FP8_E4M3_MAX
    quantized = (weight_reshaped / scales.unsqueeze(-1)).clamp(
        -FP8_E4M3_MAX, FP8_E4M3_MAX
    )
    quantized = quantized.view(vocab_size, hidden_size)
    embedding_fp8 = quantized.to(torch.float8_e4m3fn)
    logger.info(
        "Quantized embedding to FP8: [%s, %s], scales: [%s, %s]",
        vocab_size,
        hidden_size,
        vocab_size,
        num_groups,
    )
    return embedding_fp8, scales


def _extract_embedding_weight(embed_tokens: nn.Module) -> torch.Tensor:
    if isinstance(embed_tokens, nn.Embedding):
        return embed_tokens.weight.detach()
    weight = getattr(embed_tokens, "weight", None)
    if weight is None:
        raise TypeError(
            f"Unsupported embed_tokens module: {type(embed_tokens).__name__}"
        )
    return weight.detach()


def sidecar_path(export_dir: str | Path) -> Path:
    return Path(export_dir) / LANG_EMBEDDING_SIDECAR_FILENAME


def meta_path(export_dir: str | Path) -> Path:
    return Path(export_dir) / LANG_EMBEDDING_META_FILENAME


def save_lang_embedding_sidecar(
    weight: torch.Tensor,
    export_dir: str | Path,
    *,
    filename: str = LANG_EMBEDDING_SIDECAR_FILENAME,
) -> LangEmbeddingMeta:
    """Write FP8 embedding sidecar + meta JSON under ``export_dir``."""
    from safetensors.torch import save_file

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    embedding_fp8, scales = quantize_embedding_to_fp8(weight.cpu())
    vocab_size, hidden_size = embedding_fp8.shape
    meta = LangEmbeddingMeta(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        block_size=FP8_EMBEDDING_BLOCK_SIZE,
    )

    sidecar_file = export_dir / filename
    save_file(
        {
            EMBEDDING_TENSOR_KEY: embedding_fp8,
            EMBEDDING_SCALE_TENSOR_KEY: scales,
        },
        str(sidecar_file),
    )
    with open(export_dir / LANG_EMBEDDING_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    saved = meta.saved_bytes_vs_fp16()
    pct = 100.0 * saved / max(meta.fp16_baseline_bytes(), 1)
    logger.info(
        "Saved FP8 lang embedding sidecar to %s (saved %.1f MiB, %.1f%% vs FP16)",
        sidecar_file,
        saved / (1024 * 1024),
        pct,
    )
    print(
        f"[fp8-lang-emb] sidecar={sidecar_file} "
        f"table={meta.total_bytes / (1024 * 1024):.1f} MiB "
        f"(fp16 baseline {meta.fp16_baseline_bytes() / (1024 * 1024):.1f} MiB, "
        f"saved {pct:.1f}%)"
    )
    return meta


def save_lang_embedding_sidecar_from_module(
    embed_tokens: nn.Module,
    export_dir: str | Path,
) -> LangEmbeddingMeta:
    weight = _extract_embedding_weight(embed_tokens)
    return save_lang_embedding_sidecar(weight, export_dir)


def load_lang_embedding_sidecar(
    engine_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    filename: str = LANG_EMBEDDING_SIDECAR_FILENAME,
) -> Fp8LangEmbeddingTables:
    """Load FP8 lang embedding sidecar from ``engine_dir``."""
    from safetensors.torch import load_file

    engine_dir = Path(engine_dir)
    sidecar_file = engine_dir / filename
    if not sidecar_file.is_file():
        raise FileNotFoundError(f"FP8 lang embedding sidecar not found: {sidecar_file}")

    meta_file = engine_dir / LANG_EMBEDDING_META_FILENAME
    if meta_file.is_file():
        with open(meta_file, encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = LangEmbeddingMeta(**meta_dict)
    else:
        tensors_cpu = load_file(str(sidecar_file), device="cpu")
        emb = tensors_cpu[EMBEDDING_TENSOR_KEY]
        meta = LangEmbeddingMeta(
            vocab_size=int(emb.shape[0]),
            hidden_size=int(emb.shape[1]),
            block_size=FP8_EMBEDDING_BLOCK_SIZE,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tensors = load_file(str(sidecar_file), device=str(device))
    if EMBEDDING_TENSOR_KEY not in tensors:
        raise KeyError(
            f"'{EMBEDDING_TENSOR_KEY}' missing in {sidecar_file}; keys={list(tensors)}"
        )
    if EMBEDDING_SCALE_TENSOR_KEY not in tensors:
        raise KeyError(
            f"'{EMBEDDING_SCALE_TENSOR_KEY}' missing in {sidecar_file}; keys={list(tensors)}"
        )

    return Fp8LangEmbeddingTables(
        embedding=tensors[EMBEDDING_TENSOR_KEY],
        scales=tensors[EMBEDDING_SCALE_TENSOR_KEY],
        meta=meta,
        device=device,
    )


def sidecar_exists(engine_dir: str | Path) -> bool:
    return sidecar_path(engine_dir).is_file()


def resolve_use_fp8_lang_embedding(config: Any | None, engine_dir: str | Path) -> bool:
    """Resolve whether to use FP8 lang embedding sidecar at runtime."""
    import os

    env = os.environ.get("MO_FP8_LANG_EMB", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        if not sidecar_exists(engine_dir):
            raise FileNotFoundError(
                f"MO_FP8_LANG_EMB=1 but sidecar missing under {engine_dir}"
            )
        return True

    if config is not None:
        raw = getattr(config, "use_fp8_lang_embedding", None)
        if raw is None and isinstance(config, dict):
            raw = config.get("use_fp8_lang_embedding")
        if raw is not None:
            enabled = bool(raw)
            if enabled and not sidecar_exists(engine_dir):
                raise FileNotFoundError(
                    f"use_fp8_lang_embedding=True but sidecar missing under {engine_dir}"
                )
            return enabled

    return sidecar_exists(engine_dir)
