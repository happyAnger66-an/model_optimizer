# Gemma fused gated MLP TRT plugin (fp16/bf16 GeGLU / gelu_pytorch_tanh).
#
# TensorRT_ROOT: same resolution as siglip_ffn_fp8 (SDK or triplet include dir).

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED ENV{TENSORRT_ROOT} AND NOT TensorRT_ROOT)
  message(FATAL_ERROR "Set TensorRT_ROOT (or env TENSORRT_ROOT).")
endif()
if(DEFINED ENV{TENSORRT_ROOT})
  set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root" FORCE)
endif()

get_filename_component(_tensorrt_root_norm "${TensorRT_ROOT}" ABSOLUTE)
if(EXISTS "${_tensorrt_root_norm}/include/NvInfer.h")
  set(TRT_INCLUDE "${_tensorrt_root_norm}/include")
elseif(EXISTS "${_tensorrt_root_norm}/NvInfer.h")
  set(TRT_INCLUDE "${_tensorrt_root_norm}")
else()
  message(FATAL_ERROR "NvInfer.h not found under TensorRT_ROOT=${_tensorrt_root_norm}")
endif()

set(_nvinfer_hints "${_tensorrt_root_norm}/lib" "${_tensorrt_root_norm}/lib/x64" "${_tensorrt_root_norm}/lib/stubs")
if(DEFINED CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND _nvinfer_hints "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()
list(APPEND _nvinfer_hints /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)

find_library(NVINFER_LIB nvinfer HINTS ${_nvinfer_hints})
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "libnvinfer not found.")
endif()

add_library(trt_gemma_fused_gated_mlp_plugin SHARED
  gemma_fused_gated_mlp_plugin.cpp
  gemma_fused_gated_mlp_cuda.cu
)

set_target_properties(trt_gemma_fused_gated_mlp_plugin PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(trt_gemma_fused_gated_mlp_plugin PRIVATE
  "${TRT_INCLUDE}"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(trt_gemma_fused_gated_mlp_plugin PRIVATE
  CUDA::cudart
  CUDA::cublasLt
  "${NVINFER_LIB}"
)

target_compile_options(trt_gemma_fused_gated_mlp_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET trt_gemma_fused_gated_mlp_plugin PROPERTY CUDA_ARCHITECTURES native)
endif()

message(STATUS "trt_gemma_fused_gated_mlp_plugin: TensorRT_ROOT=${TensorRT_ROOT} NVINFER_LIB=${NVINFER_LIB}")
