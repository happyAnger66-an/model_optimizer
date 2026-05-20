#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 将 OpenPI JAX（Orbax）checkpoint 转为 PyTorch ``model.safetensors``，供
# ``serve_policy`` / ``create_trained_policy`` 等 PyTorch 推理路径使用。
#
# 示例：
#
#   # 注册名（与 serve_policy 一致）
#   python scripts/deployment/pi05/jax2pytorch.py \\
#     --config-name pi05_libero \\
#     --checkpoint-dir /path/to/jax_checkpoint \\
#     --output-path /path/to/pytorch_checkpoint
#
#   # 自定义 TrainConfig 模块
#   python scripts/deployment/pi05/jax2pytorch.py \\
#     --config-name /path/to/my_train_config.py \\
#     --checkpoint-dir /path/to/jax_checkpoint \\
#     --output-path /path/to/out --precision bfloat16
#
#   # 仅查看 JAX 参数树
#   python scripts/deployment/pi05/jax2pytorch.py \\
#     --config-name pi05_libero \\
#     --checkpoint-dir /path/to/jax_checkpoint --inspect-only
#
# 依赖：``openpi``、``orbax-checkpoint``、``safetensors``、``torch``、``tyro``。

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import tyro

from model_optimizer.convert.jax_to_pytorch import run_jax_to_pytorch


@dataclasses.dataclass
class Args:
    """JAX → PyTorch 权重转换（对齐 openpi ``convert_jax_model_to_pytorch``）。"""

    checkpoint_dir: str
    """JAX checkpoint 目录（内含 ``params/``）。"""

    config_name: str = "pi05_libero"
    """``get_config`` 注册名，或含 ``cfg = TrainConfig(...)`` 的 ``.py`` 路径（同 ``serve_policy``）。"""

    output_path: str | None = None
    """输出目录；写入 ``model.safetensors`` 与 ``config.json``。转换时必填。"""

    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    """保存的 PyTorch 权重精度。"""

    inspect_only: bool = False
    """为 True 时仅打印 Orbax 参数树，不写 PyTorch 文件。"""


def main(args: Args) -> None:
    run_jax_to_pytorch(
        args.checkpoint_dir,
        args.config_name,
        output_path=args.output_path,
        precision=args.precision,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
