#!/usr/bin/env python3
# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""AOT-compile CuTe DSL kernels into a static library for CMake linking.

model_optimizer 独立 CuTe DSL 构建编排器（不依赖 TensorRT-Edge-LLM）。

Usage (from repo root):
  python kernelSrc/build_cutedsl.py
  python kernelSrc/build_cutedsl.py --kernels fmha --gpu_arch sm_110 --clean
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = (_SCRIPT_DIR / "../cpp/kernels/cuteDSLArtifact").resolve()
_CUTLASS_DSL_VERSION = "4.4.1"
_CUPY_VERSIONS = {12: ("cupy-cuda12x", "12.3.0"), 13: ("cupy-cuda13x", "13.6.0")}

_LLM_HOMO = [
    "--export_only",
    "--is_causal",
    "--is_persistent",
    "--bottom_right_align",
]


@dataclass
class KernelVariant:
    name: str
    group: str
    supported_sms: list[int]
    script: str
    script_args: list[str] = field(default_factory=list)


KERNEL_VARIANTS = [
  KernelVariant(
        name="fmha_d256_homo_bf16",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_d256_cutedsl/export.py",
        script_args=[
            "--q_shape", "1,128,8,256",
            "--kv_cap", "4096",
            "--in_dtype", "BFloat16",
        ] + _LLM_HOMO,
    ),
    KernelVariant(
        name="fmha_d256_homo_fp16",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_d256_cutedsl/export.py",
        script_args=[
            "--q_shape", "1,128,8,256",
            "--kv_cap", "4096",
            "--in_dtype", "Float16",
        ] + _LLM_HOMO,
    ),
]

_ALL_GROUPS = {v.group for v in KERNEL_VARIANTS}


def _parse_sm(gpu_arch_str: str) -> int:
    s = gpu_arch_str.strip().lower()
    if s.startswith("sm_"):
        s = s[3:]
    sm = int(s)
    if sm <= 0:
        raise ValueError(f"Invalid SM: {gpu_arch_str}")
    return sm


def detect_gpu_sm() -> int:
    try:
        import cupy

        return int(cupy.cuda.Device(0).compute_capability)
    except Exception:
        pass
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not detect GPU SM; pass --gpu_arch")
    line = result.stdout.strip().splitlines()[0].strip()
    major, minor = line.split(".")
    return int(major) * 10 + int(minor)


def select_variants(sm: int, kernels_arg: str) -> list[KernelVariant]:
    groups_requested = kernels_arg.strip().upper()
    if groups_requested == "ALL":
        return [v for v in KERNEL_VARIANTS if sm in v.supported_sms]
    tokens = [t.strip().lower() for t in kernels_arg.split(",")]
    unknown = [t for t in tokens if t not in _ALL_GROUPS]
    if unknown:
        raise ValueError(f"Unknown kernel group(s): {unknown}. Valid: {sorted(_ALL_GROUPS)}")
    selected = [v for v in KERNEL_VARIANTS if v.group in tokens and sm in v.supported_sms]
    if not selected:
        raise ValueError(f"No variants in {tokens} support SM{sm}")
    return selected


def detect_arch(override: str | None) -> str:
    if override:
        m = override.lower().replace("-", "_")
        if m in ("x86_64", "amd64"):
            return "x86_64"
        if m in ("aarch64", "arm64"):
            return "aarch64"
        raise ValueError(f"Unsupported --arch: {override}")
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(f"Unsupported architecture: {platform.machine()}")


def sm_to_artifact_tag(sm: int) -> str:
    return f"sm_{sm}"


def _nvcc_version() -> str | None:
    for nvcc in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            for token in out.split():
                if token.startswith("V") and token[1:2].isdigit():
                    return token[1:].split(",")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    try:
        import cupy

        v = cupy.cuda.runtime.runtimeGetVersion()
        major, rest = divmod(v, 1000)
        minor, patch = divmod(rest, 10)
        return f"{major}.{minor}.{patch}"
    except Exception:
        return None


def check_dependencies() -> tuple[str, Path, str]:
    errors: list[str] = []
    try:
        ver = importlib.metadata.version("nvidia-cutlass-dsl")
        if ver != _CUTLASS_DSL_VERSION:
            errors.append(
                f"nvidia-cutlass-dsl: found {ver}, need {_CUTLASS_DSL_VERSION}"
            )
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        pkg_dir = (
            Path(next(iter(spec.submodule_search_locations)))
            if spec.submodule_search_locations
            else Path(spec.origin).parent
        )
        lib_dir = pkg_dir / "lib"
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"nvidia-cutlass-dsl not found (need {_CUTLASS_DSL_VERSION})")
        lib_dir, ver = None, "unknown"

    cuda_ver = _nvcc_version()
    if not cuda_ver:
        errors.append("Could not detect CUDA version")
    if not shutil.which("ar"):
        errors.append("'ar' not found on PATH")

    if errors:
        print("Dependency check failed:\n" + "\n".join(f"  • {e}" for e in errors))
        sys.exit(1)

    assert ver is not None and lib_dir is not None and cuda_ver is not None
    print(f"  nvidia-cutlass-dsl=={ver} ✓  CUDA {cuda_ver} ✓")
    return ver, lib_dir, cuda_ver


def _compile_one(variant: KernelVariant, staging_dir: Path, verbose: bool):
    cmd = [sys.executable, str(_SCRIPT_DIR / variant.script)]
    cmd += [
        "--output_dir",
        str(staging_dir),
        "--file_name",
        variant.name,
        "--function_prefix",
        variant.name,
    ]
    cmd += variant.script_args
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR), capture_output=not verbose, text=True)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        return variant.name, False, elapsed, (result.stderr or result.stdout or "")[:4000]
    obj = staging_dir / f"{variant.name}.o"
    hdr = staging_dir / f"{variant.name}.h"
    if not obj.exists() or not hdr.exists():
        return variant.name, False, elapsed, f"Missing {obj.name} or {hdr.name}"
    return variant.name, True, elapsed, ""


def compile_variants(variants, staging_dirs, jobs, verbose):
    print(f"\nCompiling {len(variants)} variant(s) (jobs={jobs})...")
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_compile_one, v, staging_dirs[v.name], verbose): v for v in variants
        }
        for future in concurrent.futures.as_completed(futures):
            name, ok, elapsed, msg = future.result()
            print(f"  {'✓' if ok else '✗'} {name:<30} ({elapsed:.1f}s)")
            if not ok:
                failures.append((name, msg))
    if failures:
        for name, msg in failures:
            print(f"\n  [{name}]\n{msg}")
        sys.exit(1)


def build(args) -> None:
    sm = _parse_sm(args.gpu_arch) if args.gpu_arch else detect_gpu_sm()
    arch = detect_arch(args.arch)
    artifact_tag = sm_to_artifact_tag(sm)
    output_dir = Path(args.output_dir) / arch / artifact_tag

    print(f"Target arch : {arch}")
    print(f"GPU SM      : SM{sm}")
    print(f"Output dir  : {output_dir}")

    variants = select_variants(sm, args.kernels)
    if not variants:
        print("No variants selected.")
        return

    print(f"Variants    : {[v.name for v in variants]}")
    print("\nChecking dependencies...")
    dsl_ver, lib_dir, cuda_ver = check_dependencies()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    root_staging = Path(tempfile.mkdtemp(prefix="modelopt_cutedsl_"))
    try:
        staging_dirs = {}
        for v in variants:
            d = root_staging / v.name
            d.mkdir()
            staging_dirs[v.name] = d

        compile_variants(variants, staging_dirs, args.jobs, args.verbose)

        kernel_objs = [staging_dirs[v.name] / f"{v.name}.o" for v in variants]
        runtime = lib_dir / "libcuda_dialect_runtime_static.a"
        if not runtime.exists():
            raise FileNotFoundError(f"{runtime} not found")

        runtime_obj_dir = root_staging / "runtime_objs"
        runtime_obj_dir.mkdir()
        subprocess.run(["ar", "x", str(runtime)], cwd=str(runtime_obj_dir), check=True)
        runtime_objs = sorted(runtime_obj_dir.glob("*.o"))

        output_dir.mkdir(parents=True, exist_ok=True)
        lib_path = output_dir / f"libcutedsl_{arch}.a"
        subprocess.run(
            ["ar", "rcs", str(lib_path)]
            + [str(o) for o in kernel_objs]
            + [str(o) for o in runtime_objs],
            check=True,
        )
        print(f"\n  Created {lib_path.name} ({lib_path.stat().st_size // 1024} KB)")

        inc_dir = output_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        for v in variants:
            shutil.copy2(staging_dirs[v.name] / f"{v.name}.h", inc_dir)

        groups = sorted({v.group for v in variants})
        for group in groups:
            group_vars = [v for v in variants if v.group == group]
            (inc_dir / f"cutedsl_{group}_all.h").write_text(
                "#pragma once\n"
                + "".join(f'#include "{v.name}.h"\n' for v in group_vars)
            )
        (inc_dir / "cutedsl_all.h").write_text(
            "#pragma once\n" + "".join(f'#include "{v.name}.h"\n' for v in variants)
        )

        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "arch": arch,
                    "artifact_tag": artifact_tag,
                    "gpu_arch": f"sm_{sm}",
                    "cuda_version": cuda_ver,
                    "cutlass_dsl_version": dsl_ver,
                    "build_date": datetime.now(timezone.utc).isoformat(),
                    "groups": groups,
                    "variants": [v.name for v in variants],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        shutil.rmtree(root_staging, ignore_errors=True)

    print(f"\nDone. Artifacts: {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu_arch", default=None, help="Target SM, e.g. sm_110")
    p.add_argument("--kernels", default="ALL", help="ALL or fmha")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    p.add_argument("--arch", default=None, help="x86_64 or aarch64")
    p.add_argument("-j", "--jobs", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--clean", action="store_true")
    build(p.parse_args())


if __name__ == "__main__":
    main()
