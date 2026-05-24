# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
#
# Simplified CuTe DSL CMake integration for model_optimizer.
# Artifacts: cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/

set(ENABLE_CUTE_DSL "OFF" CACHE STRING "CuTe DSL: OFF, ALL, or fmha")
set(CUTE_DSL_ARTIFACT_TAG "" CACHE STRING "Artifact tag e.g. sm_110")

if(DEFINED _MODEL_OPT_CUTE_DSL_INCLUDED)
  return()
endif()
set(_MODEL_OPT_CUTE_DSL_INCLUDED TRUE)

function(cute_dsl_setup)
  cmake_parse_arguments(CUTE_DSL "" "" "TARGETS;LINK_TARGETS" ${ARGN})

  if(ENABLE_CUTE_DSL STREQUAL "OFF")
    message(STATUS "CuTe DSL disabled (ENABLE_CUTE_DSL=OFF)")
    return()
  endif()

  set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
  if(_arch MATCHES "aarch64|ARM64")
    set(_host_arch "aarch64")
  else()
    set(_host_arch "x86_64")
  endif()

  set(_artifact_tag "${CUTE_DSL_ARTIFACT_TAG}")
  if(_artifact_tag STREQUAL "")
    file(GLOB _tags RELATIVE
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuteDSLArtifact/${_host_arch}"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuteDSLArtifact/${_host_arch}/*")
    list(LENGTH _tags _n)
    if(_n EQUAL 1)
      set(_artifact_tag "${_tags}")
    else()
      message(FATAL_ERROR
        "Set -DCUTE_DSL_ARTIFACT_TAG=sm_XXX (found ${_n} tags under cuteDSLArtifact/${_host_arch})")
    endif()
  endif()

  set(_artifact_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuteDSLArtifact/${_host_arch}/${_artifact_tag}")

  if(NOT EXISTS "${_artifact_dir}/metadata.json")
    message(FATAL_ERROR
      "CuTe DSL artifacts not found at ${_artifact_dir}. "
      "Run: python kernelSrc/build_cutedsl.py --gpu_arch ${_artifact_tag}")
  endif()

  set(_lib "${_artifact_dir}/libcutedsl_${_host_arch}.a")
  if(NOT EXISTS "${_lib}")
    message(FATAL_ERROR "Missing ${_lib}")
  endif()

  set(_inc "${_artifact_dir}/include")
  message(STATUS "CuTe DSL artifact: ${_artifact_dir}")

  foreach(_tgt IN LISTS CUTE_DSL_TARGETS CUTE_DSL_LINK_TARGETS)
    if(NOT TARGET ${_tgt})
      continue()
    endif()
    target_include_directories(${_tgt} PUBLIC "${_inc}")
    target_compile_definitions(${_tgt} PUBLIC CUTE_DSL_FMHA_ENABLED=1)
    if(${_tgt} IN_LIST CUTE_DSL_LINK_TARGETS)
      target_link_libraries(${_tgt} PUBLIC "${_lib}")
    endif()
  endforeach()
endfunction()
