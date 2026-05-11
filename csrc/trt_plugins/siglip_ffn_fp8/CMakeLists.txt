# SigLIP encoder FFN FP8 fused plugin (FlashRT-compatible path).
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
if(NOT EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cuda_fp8.h")
  message(FATAL_ERROR "SigLIP FP8 TRT plugin needs cuda_fp8.h (CUDA 12.8+). CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}. Set CUDAToolkit_ROOT to a newer toolkit.")
endif()

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

add_library(trt_siglip_ffn_fp8_plugin SHARED
  siglip_ffn_fp8_plugin.cpp
  siglip_ffn_fp8_gelu_bias_res.cu
)

set_target_properties(trt_siglip_ffn_fp8_plugin PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(trt_siglip_ffn_fp8_plugin PRIVATE
  "${TRT_INCLUDE}"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(trt_siglip_ffn_fp8_plugin PRIVATE
  CUDA::cudart
  CUDA::cublasLt
  "${NVINFER_LIB}"
)

target_compile_options(trt_siglip_ffn_fp8_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET trt_siglip_ffn_fp8_plugin PROPERTY CUDA_ARCHITECTURES native)
endif()

message(STATUS "trt_siglip_ffn_fp8_plugin: TensorRT_ROOT=${TensorRT_ROOT} NVINFER_LIB=${NVINFER_LIB}")
