"""按 ``.so`` 路径装载 FlashRT 编译出的 kernel 扩展（``fvk``），**不 import flash_rt 包**。

FlashRT 的算子（``gmm_fp16`` / ``fp8_gemm_descale_fp16`` / ``fused_adarms_fp8_static_fp16``
/ ``attention_qkv_fp16`` …）由其 ``csrc`` 在 Thor(SM110) 上 cmake 构建为一个
``flash_rt_kernels*.so`` 扩展。``fp8_gemm_descale_fp16`` 语义见
``docs/optimizer/flashrt/fp8_gemm_descale_fp16.md``。该 CUDA 代码量约 27MB/700+ 文件（含 CUTLASS FMHA / FP8 GEMM），
不可能在仓内重写，故本仓库**复用其编译产物**，但只按文件路径动态加载，不依赖 ``flash_rt`` 的
任何 python 模块。

构建（在 Thor 上，FlashRT 源码树内）::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target flash_rt_kernels
    # 产出 build/flash_rt_kernels*.so（及 libfmha_fp16_strided.so）

加载顺序：
  1) 环境变量 ``MO_FLASHRT_KERNELS_SO``（显式 .so 全路径）；
  2) 环境变量 ``MO_FLASHRT_BUILD_DIR`` 下的 ``flash_rt_kernels*.so``；
  3) 传入的 ``build_dir`` 参数。
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

_KERNELS_CACHE: ModuleType | None = None
_KERNELS_SO_STEM = "flash_rt_kernels"
_FMHA_SO_NAME = "libfmha_fp16_strided.so"


def _is_kernels_so(path: str) -> bool:
    """True 若 ``path`` 为 pybind 扩展 ``flash_rt_kernels*.so``（非 libfmha）。"""
    return Path(path).name.split(".", 1)[0].startswith(_KERNELS_SO_STEM)


def _find_fmha_so(build_dir: str | None, explicit: str | None) -> str | None:
    """解析 ``libfmha_fp16_strided.so``（dlopen，非 Python 模块）。"""
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())
    env_fmha = os.environ.get("MO_FLASHRT_FMHA_SO", "").strip()
    if env_fmha and Path(env_fmha).is_file():
        return str(Path(env_fmha).resolve())
    search_dirs = [
        os.environ.get("MO_FLASHRT_BUILD_DIR", "").strip(),
        build_dir or "",
    ]
    for d in search_dirs:
        if not d:
            continue
        base = Path(d).expanduser()
        for pattern in (_FMHA_SO_NAME, f"**/{_FMHA_SO_NAME}"):
            for hit in sorted(glob.glob(str(base / pattern), recursive="**" in pattern)):
                if Path(hit).is_file():
                    return str(Path(hit).resolve())
    return None


def _find_kernel_so(build_dir: str | None) -> str:
    candidates: list[str] = []
    env_so = os.environ.get("MO_FLASHRT_KERNELS_SO", "").strip()
    if env_so:
        candidates.append(env_so)
    search_dirs = [
        os.environ.get("MO_FLASHRT_BUILD_DIR", "").strip(),
        build_dir or "",
    ]
    for d in search_dirs:
        if not d:
            continue
        candidates.extend(sorted(glob.glob(str(Path(d).expanduser() / "flash_rt_kernels*.so"))))
        candidates.extend(
            sorted(glob.glob(str(Path(d).expanduser() / "**" / "flash_rt_kernels*.so"), recursive=True))
        )
    wrong_fmha: list[str] = []
    for c in candidates:
        if not c or not Path(c).is_file():
            continue
        resolved = str(Path(c).resolve())
        if _is_kernels_so(resolved):
            return resolved
        if Path(resolved).name.startswith("libfmha"):
            wrong_fmha.append(resolved)
    if wrong_fmha:
        raise FileNotFoundError(
            "MO_FLASHRT_KERNELS_SO 指向了 libfmha_fp16_strided.so，它不是 Python 扩展。\n"
            "请改为：\n"
            "  export MO_FLASHRT_KERNELS_SO=/path/to/flash_rt_kernels.cpython-*.so\n"
            "  export MO_FLASHRT_FMHA_SO=/path/to/libfmha_fp16_strided.so   # 可选，仅 SigLIP\n"
            f"  (错误路径: {wrong_fmha[0]})"
        )
    raise FileNotFoundError(
        "未找到 flash_rt_kernels*.so。请在 Thor 上构建后通过 MO_FLASHRT_KERNELS_SO "
        "或 MO_FLASHRT_BUILD_DIR 指定，例如：\n"
        "  cmake --build build -j --target flash_rt_kernels\n"
        "  export MO_FLASHRT_BUILD_DIR=/path/to/FlashRT/build\n"
        f"  (已搜索: {[c for c in candidates if c]})"
    )


def load_kernels(build_dir: str | None = None, *, fmha_so: str | None = None) -> ModuleType:
    """加载 ``flash_rt_kernels`` 扩展模块（缓存单例）。

    Args:
        build_dir: 可选 FlashRT cmake build 目录。
        fmha_so: 可选 ``libfmha_fp16_strided.so`` 路径；提供时调用
            ``fvk.load_fmha_strided_library`` 注册 CUTLASS FMHA。

    Returns:
        kernel 扩展模块（即 FlashRT 中的 ``fvk``）。
    """
    global _KERNELS_CACHE
    if _KERNELS_CACHE is not None:
        return _KERNELS_CACHE

    so_path = _find_kernel_so(build_dir)
    # C 扩展的初始化符号在编译期固定为 ``PyInit_<模块名最后一段>``；FlashRT 的扩展按
    # ``flash_rt.flash_rt_kernels`` 编译，导出 ``PyInit_flash_rt_kernels``。因此按 ``.so``
    # 文件名词干（首个 ``.`` 之前，如 ``flash_rt_kernels.cpython-310-...-gnu.so`` →
    # ``flash_rt_kernels``）作为模块名加载，使 CPython 能匹配到 init 符号。
    # 注意：这只在 ``sys.modules`` 注册顶层名 ``flash_rt_kernels``，**不会** import
    # ``flash_rt`` python 包（其子模块名为 ``flash_rt.flash_rt_kernels``，不冲突）。
    mod_name = Path(so_path).name.split(".", 1)[0] or "flash_rt_kernels"
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {so_path} 创建扩展模块 spec")
    mod = importlib.util.module_from_spec(spec)
    # 单相初始化的扩展可能在 init 期间自引用，需先登记到 sys.modules 再 exec。
    sys.modules.setdefault(mod_name, mod)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # 加载失败时回收占位，避免污染后续 import。
        if sys.modules.get(mod_name) is mod:
            del sys.modules[mod_name]
        raise

    fmha = _find_fmha_so(build_dir, fmha_so)
    if fmha and hasattr(mod, "load_fmha_strided_library"):
        ret = mod.load_fmha_strided_library(fmha)
        if ret != 0:
            logger.warning(
                "[flashrt-kernels] load_fmha_strided_library failed (%s), ret=%s; "
                "decoder attention 仍可用 fvk 内置路径",
                fmha,
                ret,
            )

    _KERNELS_CACHE = mod
    return mod
