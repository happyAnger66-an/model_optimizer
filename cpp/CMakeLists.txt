# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(model_opt_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(TRT_PACKAGE_DIR "" CACHE PATH "TensorRT package root, include dir, or lib dir")

find_package(CUDAToolkit REQUIRED)

find_path(NVINFER_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    ${TRT_PACKAGE_DIR}/include
    ${TRT_PACKAGE_DIR}
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/local/include
)

find_library(NVINFER_LIBRARY
  NAMES nvinfer
  HINTS
    ${TRT_PACKAGE_DIR}/lib
    ${TRT_PACKAGE_DIR}/lib64
    ${TRT_PACKAGE_DIR}
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
)

find_library(NVINFER_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS
    ${TRT_PACKAGE_DIR}/lib
    ${TRT_PACKAGE_DIR}/lib64
    ${TRT_PACKAGE_DIR}
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
)

if(NOT NVINFER_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
  message(FATAL_ERROR
    "TensorRT not found. Set -DTRT_PACKAGE_DIR to the TensorRT root, include dir, "
    "or lib dir. Found include='${NVINFER_INCLUDE_DIR}', lib='${NVINFER_LIBRARY}'")
endif()

message(STATUS "TensorRT include: ${NVINFER_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${NVINFER_LIBRARY}")
if(NVINFER_PLUGIN_LIBRARY)
  message(STATUS "TensorRT nvinfer_plugin: ${NVINFER_PLUGIN_LIBRARY}")
endif()

include(cmake/CuteDsl.cmake)

add_library(model_opt_kernels STATIC
  kernels/fmha/cuteDslFMHAD256Runner.cpp
)

target_include_directories(model_opt_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(model_opt_kernels PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
)

cute_dsl_setup(
  TARGETS model_opt_kernels
  LINK_TARGETS model_opt_kernels
)

add_library(model_opt_plugin SHARED
  plugins/fmha_d256/FmhaD256AttentionPlugin.cpp
  plugins/pluginEntry.cpp
)

target_include_directories(model_opt_plugin PRIVATE
  ${NVINFER_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/plugins
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(model_opt_plugin PRIVATE
  model_opt_kernels
  ${NVINFER_LIBRARY}
)

if(NVINFER_PLUGIN_LIBRARY)
  target_link_libraries(model_opt_plugin PRIVATE ${NVINFER_PLUGIN_LIBRARY})
endif()

set_target_properties(model_opt_plugin PROPERTIES
  OUTPUT_NAME model_opt_plugin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
