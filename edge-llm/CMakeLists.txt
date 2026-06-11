cmake_minimum_required(VERSION 3.18)
project(edge_llm_pi05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# TensorRT: 优先 TRT_PACKAGE_DIR（与 TensorRT-Edge-LLM 约定一致），否则查系统路径。
if(DEFINED TRT_PACKAGE_DIR)
  set(TENSORRT_INCLUDE_DIR "${TRT_PACKAGE_DIR}/include")
  set(TENSORRT_LIB_DIR "${TRT_PACKAGE_DIR}/lib")
  find_library(TENSORRT_LIB_NVINFER nvinfer PATHS "${TENSORRT_LIB_DIR}" NO_DEFAULT_PATH)
else()
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/TensorRT/include)
  find_library(TENSORRT_LIB_NVINFER nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
endif()

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB_NVINFER)
  message(FATAL_ERROR "TensorRT not found. Set -DTRT_PACKAGE_DIR=/path/to/TensorRT")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_LIB_NVINFER}")

# sentencepiece（PaliGemma tokenizer）：静态构建，避免运行时依赖。
include(FetchContent)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sentencepiece)

add_library(pi05_runtime STATIC
  src/common/logger.cpp
  src/common/dtype.cpp
  src/common/tensor.cpp
  src/common/npy.cpp
  src/runtime/trtEngine.cpp
  src/pi05/pi05Runtime.cpp
  src/pi05/assets.cpp
  src/pi05/tokenizer.cpp
  src/pi05/preprocess.cpp
)
target_include_directories(pi05_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${TENSORRT_INCLUDE_DIR}
)
target_include_directories(pi05_runtime PRIVATE
  ${sentencepiece_SOURCE_DIR}/src
)
target_link_libraries(pi05_runtime PUBLIC ${TENSORRT_LIB_NVINFER} CUDA::cudart
  PRIVATE sentencepiece-static)
target_compile_options(pi05_runtime PRIVATE -Wall -Wextra)

add_executable(pi05_compare src/pi05/pi05CompareMain.cpp)
target_link_libraries(pi05_compare PRIVATE pi05_runtime)
target_compile_options(pi05_compare PRIVATE -Wall -Wextra)

add_executable(pi05_infer src/pi05/pi05InferMain.cpp)
target_link_libraries(pi05_infer PRIVATE pi05_runtime)
target_compile_options(pi05_infer PRIVATE -Wall -Wextra)
