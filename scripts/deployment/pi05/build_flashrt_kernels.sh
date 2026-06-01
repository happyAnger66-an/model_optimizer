#!/usr/bin/env bash
# 在 Thor(SM110) 上构建 FlashRT 的 kernel 扩展，供 model_optimizer 的
# native FlashRT decoder 后端按路径加载（不 import flash_rt python 包）。
#
# 用法:
#   FLASHRT_SRC=/home/zhangxa/codes/FlashRT bash build_flashrt_kernels.sh
#
# 构建完成后，让 model_optimizer 装载 kernel:
#   export MO_FLASHRT_BUILD_DIR=$FLASHRT_SRC/build
#   export MO_FLASHRT_FMHA_SO=$MO_FLASHRT_BUILD_DIR/libfmha_fp16_strided.so
set -euo pipefail

FLASHRT_SRC="${FLASHRT_SRC:-/home/zhangxa/codes/FlashRT}"
BUILD_DIR="${BUILD_DIR:-$FLASHRT_SRC/build}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "$FLASHRT_SRC" ]]; then
  echo "[build] FlashRT 源码目录不存在: $FLASHRT_SRC" >&2
  echo "[build] 设置 FLASHRT_SRC=指向 FlashRT 仓库根目录" >&2
  exit 1
fi

echo "[build] FlashRT 源: $FLASHRT_SRC"
echo "[build] build 目录: $BUILD_DIR"
echo "[build] 并行度: $JOBS"

cmake -S "$FLASHRT_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
# kernel 扩展（含 CUTLASS FP8 GEMM / fused adarms / attention 等 decoder 所需算子）
cmake --build "$BUILD_DIR" -j "$JOBS" --target flash_rt_kernels
# CUTLASS strided FMHA（SigLIP/encoder 用；decoder cuBLAS attention 可选）
cmake --build "$BUILD_DIR" -j "$JOBS" --target fmha_fp16_strided 2>/dev/null || \
  echo "[build] (可选) fmha_fp16_strided 目标未构建，decoder 走 cuBLAS attention 回退即可"

echo
echo "[build] 完成。产出:"
find "$BUILD_DIR" -maxdepth 3 -name "flash_rt_kernels*.so" -o -name "libfmha_fp16_strided.so" 2>/dev/null || true
echo
echo "[build] 让 model_optimizer 装载 kernel:"
echo "  export MO_FLASHRT_BUILD_DIR=$BUILD_DIR"
echo "  export MO_FLASHRT_FMHA_SO=$BUILD_DIR/libfmha_fp16_strided.so"
