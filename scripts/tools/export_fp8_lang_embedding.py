#!/usr/bin/env python3
"""Export FP8 language embedding sidecar from a pi0.5 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_optimizer.models.pi05.embed_prefix import Pi05EmbedPrefix
from model_optimizer.models.pi05.model_pi05 import Pi05Model
from model_optimizer.quantization.fp8_lang_embedding import save_lang_embedding_sidecar_from_module


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-config", default=None)
    args = p.parse_args()

    wrapper = Pi05Model.construct_from_name_path(
        args.model_name,
        args.model_path,
        train_config=args.train_config,
    )
    embed_prefix = Pi05EmbedPrefix.construct_model(wrapper)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = save_lang_embedding_sidecar_from_module(embed_prefix.embed_tokens, out)
    print(
        f"Wrote FP8 lang embedding sidecar to {out} "
        f"(vocab={meta.vocab_size}, hidden={meta.hidden_size}, "
        f"total={meta.total_bytes / (1024**2):.1f} MiB)"
    )


if __name__ == "__main__":
    main()
