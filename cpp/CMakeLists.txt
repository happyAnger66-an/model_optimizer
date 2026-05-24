# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(model_opt_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED TRT_PACKAGE_DIR)
  message(FATAL_ERROR "Set -DTRT_PACKAGE_DIR=/path/to/TensorRT")
endif()

find_package(CUDA REQUIRED)

include(cmake/CuteDsl.cmake)

add_library(model_opt_kernels STATIC
  kernels/fmha/cuteDslFMHAD256Runner.cpp
)

target_include_directories(model_opt_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
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
  ${TRT_PACKAGE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/plugins
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(model_opt_plugin PRIVATE
  model_opt_kernels
  ${TRT_PACKAGE_DIR}/lib/libnvinfer.so
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
)

set_target_properties(model_opt_plugin PROPERTIES
  OUTPUT_NAME model_opt_plugin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
