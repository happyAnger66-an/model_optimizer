"""Unit tests for FP8 language embedding sidecar + lookup."""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn

from model_optimizer.infer.kernels.fp8_lang_embedding import Fp8LangEmbeddingLookup
from model_optimizer.quantization.fp8_lang_embedding import (
    load_lang_embedding_sidecar,
    quantize_embedding_to_fp8,
    save_lang_embedding_sidecar,
    sidecar_exists,
)


def _tiny_weight() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(512, 256, dtype=torch.float32)


def test_quantize_shape_and_range() -> None:
    weight = _tiny_weight()
    emb_fp8, scales = quantize_embedding_to_fp8(weight)
    assert emb_fp8.shape == weight.shape
    assert emb_fp8.dtype == torch.float8_e4m3fn
    assert scales.shape == (512, 256 // 128)


def test_sidecar_roundtrip() -> None:
    weight = _tiny_weight()
    with tempfile.TemporaryDirectory() as tmp:
        meta = save_lang_embedding_sidecar(weight, tmp)
        assert sidecar_exists(tmp)
        assert meta.vocab_size == 512
        assert meta.hidden_size == 256
        tables = load_lang_embedding_sidecar(tmp, device="cpu")
        assert tables.embedding.shape == weight.shape
        assert tables.scales.shape == (512, 2)


def test_lookup_matches_fp16_baseline() -> None:
    weight = _tiny_weight()
    embedding = nn.Embedding(512, 256)
    embedding.weight.data.copy_(weight)

    with tempfile.TemporaryDirectory() as tmp:
        save_lang_embedding_sidecar(weight, tmp)
        tables = load_lang_embedding_sidecar(tmp, device="cpu")
        lookup = Fp8LangEmbeddingLookup(tables, output_dtype=torch.float32)

        token_ids = torch.tensor([[1, 2, 3, 10], [4, 5, 6, 7]], dtype=torch.long)
        expected = embedding(token_ids)
        actual = lookup(token_ids)

        cos = torch.nn.functional.cosine_similarity(
            expected.reshape(-1, 256),
            actual.reshape(-1, 256),
            dim=-1,
        )
        assert float(cos.min()) > 0.999


def test_memory_stats_savings() -> None:
    weight = _tiny_weight()
    with tempfile.TemporaryDirectory() as tmp:
        meta = save_lang_embedding_sidecar(weight, tmp)
        assert meta.saved_bytes_vs_fp16() > 0
        assert meta.total_bytes < meta.fp16_baseline_bytes()
