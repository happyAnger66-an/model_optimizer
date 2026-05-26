# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for kernelSrc/build_cutedsl.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_kernels_argv(argv: list[str] | None) -> list[str]:
    """``kernels build --config ...`` 与 ``kernels --config ...`` 等价（文档曾写 build 子命令）。"""
    if argv and argv[0] == "build":
        return argv[1:]
    return list(argv) if argv is not None else []


def kernels_build_cli(argv: list[str] | None = None) -> None:
    argv = _normalize_kernels_argv(argv)
    parser = argparse.ArgumentParser(description="Build CuTe DSL AOT kernel artifacts")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_repo_root() / "config" / "cutedsl_build.yaml"),
        help="YAML config path",
    )
    parser.add_argument("--kernels", type=str, default=None)
    parser.add_argument("--gpu_arch", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("-j", type=int, default=None)
    args = parser.parse_args(argv)

    groups = None
    gpu_arch = args.gpu_arch
    jobs = args.j
    clean = args.clean

    config_path = Path(args.config)
    if config_path.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(config_path.read_text()) or {}
            groups = args.kernels or ",".join(cfg.get("groups", ["fmha"]))
            gpu_arch = gpu_arch or cfg.get("gpu_arch")
            jobs = jobs if jobs is not None else cfg.get("jobs", 2)
            clean = clean or bool(cfg.get("clean", False))
            output_dir = cfg.get("output_dir")
        except ImportError:
            groups = args.kernels or "fmha"
            output_dir = None
    else:
        groups = args.kernels or "fmha"
        output_dir = None

    script = _repo_root() / "kernelSrc" / "build_cutedsl.py"
    cmd = [sys.executable, str(script), "--kernels", groups]
    if gpu_arch:
        cmd += ["--gpu_arch", gpu_arch]
    if jobs is not None:
        cmd += ["-j", str(jobs)]
    if clean:
        cmd.append("--clean")
    if output_dir:
        cmd += ["--output_dir", str(_repo_root() / output_dir)]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    kernels_build_cli()
