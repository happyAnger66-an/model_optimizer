# SigLIP encoder FFN FP8 fused plugin (FlashRT-compatible path).
#
# Requires CUDA + TensorRT. Example:
#   cmake -S csrc -B build_csrc \
#     -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON \
#     -DTensorRT_ROOT=/path/to/TensorRT-10.x \
#     -DCMAKE_CUDA_ARCHITECTURES=90

find_package(CUDAToolkit REQUIRED)
if(NOT EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cuda_fp8.h")
  message(FATAL_ERROR "SigLIP FP8 TRT plugin needs cuda_fp8.h (CUDA 12.8+). CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}. Set CUDAToolkit_ROOT to a newer toolkit.")
endif()

if(NOT DEFINED ENV{TENSORRT_ROOT} AND NOT TensorRT_ROOT)
  message(FATAL_ERROR "Set TensorRT_ROOT (or env TENSORRT_ROOT) to TensorRT install root.")
endif()
if(DEFINED ENV{TENSORRT_ROOT})
  set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root" FORCE)
endif()

set(TRT_INCLUDE "${TensorRT_ROOT}/include")
if(NOT EXISTS "${TRT_INCLUDE}/NvInfer.h")
  message(FATAL_ERROR "NvInfer.h not found under ${TRT_INCLUDE}")
endif()

find_library(NVINFER_LIB nvinfer HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib/x64"
  /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "libnvinfer not found under ${TensorRT_ROOT}/lib (set TensorRT_ROOT to a full TensorRT SDK).")
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
