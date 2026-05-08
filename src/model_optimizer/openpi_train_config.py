# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 与 ``serve_policy`` / ``Pi05Model`` 所用逻辑一致：``openpi`` 注册名或
# 含 ``TrainConfig`` 实例的 ``.py`` 路径，供导出 / 量化等 CLI 与 ``Pi05Model`` 使用。

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


def load_train_config(config_ref: str) -> TrainConfig:
    """加载 ``TrainConfig``：``get_config(name)`` 或从 ``.py`` 文件读取 ``cfg`` / ``config`` / ``train_config``。"""
    from openpi.training import config as _config

    path = Path(config_ref).expanduser()
    if path.is_file() and path.suffix == ".py":
        mod_name = f"_model_opt_train_cfg_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path.resolve()))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config module from {config_ref!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tc = _config.TrainConfig
        for attr in ("cfg", "config", "train_config"):
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                if isinstance(obj, tc):
                    logging.info(
                        "Loaded TrainConfig from %s (attribute %r)", path, attr
                    )
                    return obj
        raise ValueError(
            f"File {path} does not define a TrainConfig instance. "
            "Define one of: cfg, config, or train_config = TrainConfig(...)."
        )
    return _config.get_config(config_ref)
