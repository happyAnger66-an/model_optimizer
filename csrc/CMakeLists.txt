# Copyright 2025 the model_optimizer team.
#
# Optional native targets (TensorRT plugins, etc.). Default OFF so pip/source
# installs without CUDA/TensorRT still work.

cmake_minimum_required(VERSION 3.20)
project(model_optimizer_csrc LANGUAGES CXX CUDA)

option(MODEL_OPTIMIZER_BUILD_TRT_PLUGINS "Build TensorRT plugins under csrc/trt_plugins" OFF)

if(MODEL_OPTIMIZER_BUILD_TRT_PLUGINS)
  add_subdirectory(trt_plugins/siglip_ffn_fp8)
  add_subdirectory(trt_plugins/gemma_fused_gated_mlp)
endif()
