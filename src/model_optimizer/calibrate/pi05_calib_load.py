"""Pi0.5 标定数据加载：支持单文件（torch.save 的 list）或目录内分片 + manifest（低内存）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch

MANIFEST_FORMAT = "pi05_calib_shards_v1"

def _torch_load_compat(path: Path) -> Any:
    """兼容 PyTorch 2.6+ torch.load(weights_only=True 默认)。

    我们的 calib shard 是 `torch.save(list[dict[str, Tensor]])`，不属于纯 state_dict 权重文件，
    因此在 PyTorch 2.6+ 需要显式 `weights_only=False` 才能正常反序列化。
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        msg = str(exc)
        if "weights_only" in msg or "Weights only load failed" in msg:
            # 仅对本地生成的 calib 数据回退；这会允许 pickle 执行，请确保文件来源可信。
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


class Pi05CalibShardIterable:
    """按分片依次 torch.load，每次只在内存中保留当前分片。"""

    def __init__(
        self,
        *,
        shard_paths: list[Path],
        total_samples: int | None,
    ) -> None:
        self._shard_paths = shard_paths
        self._total_samples = total_samples

    def __len__(self) -> int:
        if self._total_samples is None:
            raise TypeError("shard set has no recorded total_samples (no manifest)")
        return self._total_samples

    def __iter__(self) -> Iterator[Any]:
        for path in self._shard_paths:
            chunk = _torch_load_compat(path)
            if not isinstance(chunk, list):
                chunk = list(chunk)
            try:
                yield from chunk
            finally:
                del chunk

    @classmethod
    def from_manifest(cls, save_dir: Path, manifest_path: Path) -> Pi05CalibShardIterable:
        with open(manifest_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != MANIFEST_FORMAT:
            raise ValueError(
                f"unsupported calib manifest format {meta.get('format')!r}, expect {MANIFEST_FORMAT!r}"
            )
        rels = meta.get("shard_relpaths") or []
        paths = [(save_dir / rel).resolve() for rel in rels]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"calib shard missing: {p}")
        total = meta.get("total_samples")
        if total is not None:
            total = int(total)
        return cls(shard_paths=paths, total_samples=total)

    @classmethod
    def from_shard_dir(cls, shard_dir: Path) -> Pi05CalibShardIterable:
        paths = sorted(shard_dir.glob("shard_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no shard_*.pt under {shard_dir}")
        return cls(shard_paths=[p.resolve() for p in paths], total_samples=None)


def open_pi05_calib_for_quantize(calib_data: str | os.PathLike[str], *, component: str) -> Any:
    """返回可迭代对象（list 或 Pi05CalibShardIterable），供 ``quantize_model`` / ``val`` 使用。

    - **文件** ``*.pt``：``torch.load`` 得到 list（与旧行为一致，整表进内存）。
    - **目录**（通常为 ``--calib-save-path`` 输出目录）：
      - 优先 ``{component}_calib_manifest.json`` + 其中列出的分片；
      - 否则若存在 ``{component}_calib_shards/shard_*.pt`` 则按文件名排序流式读取（无 manifest 时 ``len()`` 不可用）；
      - 否则若存在 ``{component}_calib_datas.pt`` 则整文件加载（旧合并文件）。
    - **目录** 且路径名为 ``{component}_calib_shards``：视为分片目录本身（其父目录为 save_dir，用于相对 manifest 的约定路径）。
    """
    p = Path(calib_data).expanduser().resolve()
    component = str(component)

    if p.is_file():
        data = _torch_load_compat(p)
        if not isinstance(data, list):
            data = list(data)
        return data

    if not p.is_dir():
        raise FileNotFoundError(f"calib path not found: {p}")

    if p.name == f"{component}_calib_shards":
        return Pi05CalibShardIterable.from_shard_dir(p)

    manifest = p / f"{component}_calib_manifest.json"
    if manifest.is_file():
        return Pi05CalibShardIterable.from_manifest(p, manifest)

    shard_sub = p / f"{component}_calib_shards"
    if shard_sub.is_dir() and any(shard_sub.glob("shard_*.pt")):
        return Pi05CalibShardIterable.from_shard_dir(shard_sub)

    merged = p / f"{component}_calib_datas.pt"
    if merged.is_file():
        data = _torch_load_compat(merged)
        if not isinstance(data, list):
            data = list(data)
        return data

    raise FileNotFoundError(
        f"no calib data for component {component!r} under {p}: "
        f"expect manifest, or {component}_calib_shards/shard_*.pt, or {component}_calib_datas.pt"
    )
