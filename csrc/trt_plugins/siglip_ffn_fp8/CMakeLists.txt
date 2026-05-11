# SigLIP MLP TensorRT plugin (ONNX ``trt::SiglipMlpPlugin`` / PyTorch ``trt::siglip_mlp_plugin``).
#
# Requires CUDA + TensorRT.
#
# TensorRT_ROOT supports two layouts:
#   1) SDK tarball:  <root>/include/NvInfer.h  and  <root>/lib/libnvinfer.so*
#   2) distro / sysroot triplet dir (e.g. Jetson/Thor aarch64):
#        TensorRT_ROOT=/usr/include/aarch64-linux-gnu  (NvInfer.h next to other TRT headers)
#      and libnvinfer under /usr/lib/aarch64-linux-gnu (auto-found via find_library hints).
#
# Example (x86_64 SDK):
#   cmake -S csrc -B build -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON \
#     -DTensorRT_ROOT=/path/to/TensorRT-10.x -DCMAKE_CUDA_ARCHITECTURES=90
#
# Example (aarch64 system packages, headers under triplet include):
#   cmake -S csrc -B build -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON \
#     -DTensorRT_ROOT=/usr/include/aarch64-linux-gnu \
#     -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED ENV{TENSORRT_ROOT} AND NOT TensorRT_ROOT)
  message(FATAL_ERROR "Set TensorRT_ROOT (or env TENSORRT_ROOT). See comments at top of this CMakeLists.txt.")
endif()
if(DEFINED ENV{TENSORRT_ROOT})
  set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root" FORCE)
endif()

get_filename_component(_tensorrt_root_norm "${TensorRT_ROOT}" ABSOLUTE)

# Prefer SDK layout <root>/include/NvInfer.h; else headers directly under TensorRT_ROOT (common on aarch64).
set(TRT_INCLUDE "")
if(EXISTS "${_tensorrt_root_norm}/include/NvInfer.h")
  set(TRT_INCLUDE "${_tensorrt_root_norm}/include")
elseif(EXISTS "${_tensorrt_root_norm}/NvInfer.h")
  set(TRT_INCLUDE "${_tensorrt_root_norm}")
else()
  message(FATAL_ERROR
    "NvInfer.h not found. Tried:\n"
    "  ${_tensorrt_root_norm}/include/NvInfer.h\n"
    "  ${_tensorrt_root_norm}/NvInfer.h\n"
    "Set TensorRT_ROOT to an SDK root (with include/) or to the directory that contains NvInfer.h "
    "(e.g. /usr/include/aarch64-linux-gnu on Debian/Ubuntu aarch64).")
endif()

# nvinfer: SDK lib dir first, then common multiarch locations (x86_64 / aarch64).
set(_nvinfer_hints
  "${_tensorrt_root_norm}/lib"
  "${_tensorrt_root_norm}/lib/x64"
  "${_tensorrt_root_norm}/lib/stubs"
)
if(DEFINED CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND _nvinfer_hints "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()
list(APPEND _nvinfer_hints
  /usr/lib/aarch64-linux-gnu
  /usr/lib/x86_64-linux-gnu
  /usr/local/tensorrt/lib
)

find_library(NVINFER_LIB nvinfer HINTS ${_nvinfer_hints})
if(NOT NVINFER_LIB)
  message(FATAL_ERROR
    "libnvinfer not found. Searched under TensorRT_ROOT=${_tensorrt_root_norm} and standard lib paths. "
    "Install libnvinfer or set CMAKE_PREFIX_PATH / adjust TensorRT_ROOT.")
endif()

add_library(trt_siglip_mlp_plugin SHARED
  siglip_mlp_plugin.cpp
  siglip_mlp_plugin.cu
)

set_target_properties(trt_siglip_mlp_plugin PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(trt_siglip_mlp_plugin PRIVATE
  "${TRT_INCLUDE}"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(trt_siglip_mlp_plugin PRIVATE
  CUDA::cudart
  CUDA::cublas
  "${NVINFER_LIB}"
)

target_compile_options(trt_siglip_mlp_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET trt_siglip_mlp_plugin PROPERTY CUDA_ARCHITECTURES native)
endif()

message(STATUS "trt_siglip_mlp_plugin: TensorRT_ROOT=${TensorRT_ROOT} NVINFER_LIB=${NVINFER_LIB}")

# ── FP8 FFN TensorRT plugin (FlashRT static bridge: libflash_rt_trt_bridge.a) ──
set(FLASHRT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/FlashRT" CACHE PATH
  "FlashRT source tree; build FlashRT first so flash_rt/libflash_rt_trt_bridge.a exists.")

find_library(FLASHRT_TRT_BRIDGE_LIB
  NAMES flash_rt_trt_bridge libflash_rt_trt_bridge
  PATHS "${FLASHRT_ROOT}/flash_rt"
  NO_DEFAULT_PATH
)

if(FLASHRT_TRT_BRIDGE_LIB)
  add_library(trt_siglip_ffn_fp8_flashrt_plugin SHARED siglip_ffn_fp8_flashrt_plugin.cpp)
  set_target_properties(trt_siglip_ffn_fp8_flashrt_plugin PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(trt_siglip_ffn_fp8_flashrt_plugin PRIVATE
    "${TRT_INCLUDE}"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${FLASHRT_ROOT}/csrc/trt_bridge"
  )
  target_link_libraries(trt_siglip_ffn_fp8_flashrt_plugin PRIVATE
    "${FLASHRT_TRT_BRIDGE_LIB}"
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    "${NVINFER_LIB}"
  )
  message(STATUS "trt_siglip_ffn_fp8_flashrt_plugin: FLASHRT_TRT_BRIDGE_LIB=${FLASHRT_TRT_BRIDGE_LIB}")
else()
  message(STATUS "flash_rt_trt_bridge.a not found under ${FLASHRT_ROOT}/flash_rt — skipping trt_siglip_ffn_fp8_flashrt_plugin (build third_party/FlashRT)")
endif()
