# SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""导出 pi05 C++ 推理所需的部署资产（norm stats / tokenizer / 元信息）。

C++ 侧（``pi05_infer``）不解析 JSON，资产被展开为 npy + 纯文本 meta::

    assets_out/
    ├── meta.txt                    # key=value 行
    ├── paligemma_tokenizer.model   # sentencepiece model
    ├── state_q01.npy state_q99.npy state_mean.npy state_std.npy
    └── actions_q01.npy actions_q99.npy actions_mean.npy actions_std.npy

用法::

    python scripts/prepare_pi05_assets.py \\
        --checkpoint /path/to/pytorch_pi05_libero \\
        --config-name pi05_libero \\
        --out-dir /path/to/assets_out
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument(
        "--asset-id",
        default="",
        help="checkpoint assets 子目录（缺省自动扫描 norm_stats.json）",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--output-action-dim", type=int, default=7, help="LiberoOutputs 截取维度")
    return p.parse_args()


def find_norm_stats_dir(ckpt: pathlib.Path, asset_id: str) -> pathlib.Path:
    assets = ckpt / "assets"
    if asset_id:
        d = assets / asset_id
        if (d / "norm_stats.json").is_file():
            return d
        raise FileNotFoundError(f"norm_stats.json not found under {d}")
    hits = sorted(assets.rglob("norm_stats.json"))
    if not hits:
        raise FileNotFoundError(f"no norm_stats.json under {assets}")
    if len(hits) > 1:
        print(f"warning: multiple norm_stats.json, using {hits[0]}", file=sys.stderr)
    return hits[0].parent


def main() -> int:
    args = parse_args()
    import numpy as np

    from openpi.shared import download
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()

    # ---- norm stats（与 create_trained_policy 一致：从 checkpoint assets 读） ----
    stats_dir = find_norm_stats_dir(ckpt, args.asset_id)
    norm_stats = _normalize.load(stats_dir)
    print(f"norm stats from {stats_dir}: keys={list(norm_stats)}")

    def save(name: str, arr) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}.npy shape={tuple(arr.shape)}")

    for key in ("state", "actions"):
        if key not in norm_stats:
            raise KeyError(f"norm_stats missing {key!r}")
        s = norm_stats[key]
        save(f"{key}_mean", s.mean)
        save(f"{key}_std", s.std)
        if s.q01 is None or s.q99 is None:
            raise ValueError(f"norm_stats[{key!r}] lacks q01/q99 (pi05 needs quantile norm)")
        save(f"{key}_q01", s.q01)
        save(f"{key}_q99", s.q99)

    # ---- tokenizer model（与 PaligemmaTokenizer 同源） ----
    tok_path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    )
    shutil.copyfile(tok_path, out_dir / "paligemma_tokenizer.model")
    print(f"  paligemma_tokenizer.model <- {tok_path}")

    # ---- meta.txt ----
    train_cfg = _config.get_config(args.config_name)
    model_cfg = train_cfg.model
    use_quantile = 1  # pi05: model_type != PI0
    discrete_state = int(bool(getattr(model_cfg, "discrete_state_input", False)))
    meta = {
        "config_name": args.config_name,
        "max_token_len": int(model_cfg.max_token_len),
        "action_horizon": int(model_cfg.action_horizon),
        "action_dim": int(model_cfg.action_dim),
        "output_action_dim": int(args.output_action_dim),
        "num_views": 3,
        "use_quantile_norm": use_quantile,
        "discrete_state_input": discrete_state,
        "tokenizer_model": "paligemma_tokenizer.model",
    }
    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")
    print(f"  meta.txt: {meta}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
