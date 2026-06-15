"""Runtime FP8 language embedding lookup for π0.5 hybrid embed_prefix paths."""

from __future__ import annotations

import logging
import math
import os
import types
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from termcolor import colored
from torch import Tensor

from model_optimizer.infer.perf.stage_perf import StagePerfCollector
from model_optimizer.quantization.fp8_lang_embedding import (
    FP8_EMBEDDING_BLOCK_SIZE,
    Fp8LangEmbeddingTables,
    LangEmbeddingMeta,
    load_lang_embedding_sidecar,
    resolve_use_fp8_lang_embedding,
)

logger = logging.getLogger(__name__)

KEY_EMBED_PREFIX_VISION = "embed_prefix.vision"
KEY_EMBED_PREFIX_LANG = "embed_prefix.lang_embedding"
KEY_EMBED_PREFIX_LANG_SCALE = "embed_prefix.lang_scale"
KEY_EMBED_PREFIX_VALID_TOKENS = "embed_prefix.lang_embedding.valid_tokens"
KEY_EMBED_PREFIX_BYTES_READ = "embed_prefix.lang_embedding.bytes_read"


class Fp8LangEmbeddingLookup(nn.Module):
    """token_ids [B,S] int64 → hidden [B,S,H] in ``output_dtype``."""

    def __init__(
        self,
        tables: Fp8LangEmbeddingTables,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.register_buffer("embedding", tables.embedding, persistent=False)
        self.register_buffer("scales", tables.scales, persistent=False)
        self.meta = tables.meta
        self.output_dtype = output_dtype
        self.block_size = FP8_EMBEDDING_BLOCK_SIZE
        self.hidden_size = int(tables.meta.hidden_size)
        self.num_groups = self.hidden_size // self.block_size

    def memory_stats(self) -> LangEmbeddingMeta:
        return self.meta

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        flat_ids = token_ids.reshape(-1)
        fp8_rows = self.embedding.index_select(0, flat_ids)
        scale_rows = self.scales.index_select(0, flat_ids)
        grouped = fp8_rows.view(-1, self.num_groups, self.block_size).float()
        dequant = grouped * scale_rows.unsqueeze(-1)
        out = dequant.reshape(-1, self.hidden_size).to(self.output_dtype)
        return out.view(*token_ids.shape, self.hidden_size)

    def estimate_bytes_read(self, valid_token_count: int) -> int:
        return int(valid_token_count) * self.hidden_size


def _sync_cuda_if_requested() -> None:
    if os.environ.get("MO_STAGE_PERF_CUDA_SYNC", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def embed_language_tokens_timed(
    paligemma_with_expert: Any,
    lang_tokens: Tensor,
    lang_masks: Tensor | None,
    stage_perf: StagePerfCollector | None,
    *,
    bytes_per_element: int = 1,
) -> Tensor:
    """Lookup language embeddings with optional stage_perf timing + meta."""
    ctx = (
        stage_perf.timed(KEY_EMBED_PREFIX_LANG)
        if stage_perf is not None and stage_perf.enabled
        else nullcontext()
    )
    with ctx:
        out = paligemma_with_expert.embed_language_tokens(lang_tokens)
        _sync_cuda_if_requested()

    if stage_perf is not None and stage_perf.enabled and lang_masks is not None:
        valid = int(lang_masks.sum().item())
        stage_perf.record_meta(KEY_EMBED_PREFIX_VALID_TOKENS, valid)
        stage_perf.record_meta(
            KEY_EMBED_PREFIX_BYTES_READ,
            valid * int(out.shape[-1]) * bytes_per_element,
        )
    return out


def scale_lang_embedding(
    lang_emb: Tensor,
    stage_perf: StagePerfCollector | None,
) -> Tensor:
    ctx = (
        stage_perf.timed(KEY_EMBED_PREFIX_LANG_SCALE)
        if stage_perf is not None and stage_perf.enabled
        else nullcontext()
    )
    with ctx:
        return lang_emb * math.sqrt(float(lang_emb.shape[-1]))


def maybe_install_fp8_lang_embedding(
    model: Any,
    engine_dir: str,
    config: Any | None,
    *,
    stage_perf: StagePerfCollector | None = None,
) -> Fp8LangEmbeddingLookup | None:
    """Replace ``embed_language_tokens`` when FP8 sidecar is enabled."""
    if not resolve_use_fp8_lang_embedding(config, engine_dir):
        return None

    pwe = model.paligemma_with_expert
    orig = pwe.embed_language_tokens
    if getattr(pwe, "_mopt_fp8_lang_embedding_installed", False):
        return getattr(pwe, "_mopt_fp8_lang_embedding_lookup", None)

    try:
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
    except StopIteration:
        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tables = load_lang_embedding_sidecar(engine_dir, device=device)
    lookup = Fp8LangEmbeddingLookup(tables, output_dtype=dtype).eval()

    def embed_language_tokens(token_ids: Tensor) -> Tensor:
        return lookup(token_ids)

    pwe.embed_language_tokens = embed_language_tokens
    pwe._mopt_fp8_lang_embedding_installed = True
    pwe._mopt_fp8_lang_embedding_lookup = lookup
    pwe._mopt_fp8_lang_embedding_orig = orig

    meta = lookup.memory_stats()
    saved = meta.saved_bytes_vs_fp16()
    pct = 100.0 * saved / max(meta.fp16_baseline_bytes(), 1)
    print(
        colored(
            f"FP8 lang embedding enabled from {engine_dir} "
            f"(vocab={meta.vocab_size}, hidden={meta.hidden_size}, "
            f"sidecar={meta.total_bytes / (1024 * 1024):.1f} MiB, "
            f"saved {pct:.1f}% vs FP16)",
            "green",
        )
    )
    logger.info(
        "Installed FP8 lang embedding lookup (saved %.1f MiB, %.1f%%)",
        saved / (1024 * 1024),
        pct,
    )
    return lookup


def patch_embed_prefix_with_stage_splits(
    model: Any,
    embed_prefix_fn: Any,
    stage_perf: StagePerfCollector | None,
    *,
    hook_name: str,
) -> None:
    """Wrap ``model.embed_prefix`` with outer timer + TRT hook name."""
    from model_optimizer.infer.tensorrt.trt_hook_timer import trt_hook_timer

    def wrapped(
        self_m: Any,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ):
        outer = (
            stage_perf.timed("embed_prefix")
            if stage_perf is not None and stage_perf.enabled
            else nullcontext()
        )
        with outer:
            return embed_prefix_fn(self_m, images, img_masks, lang_tokens, lang_masks)

    hooked = trt_hook_timer(hook_name)(wrapped)
    model.embed_prefix = types.MethodType(hooked, model)
