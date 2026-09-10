# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
enable_testing()

# We use rapids-cmake testing infrastructure to allow us to run multiple GPU tests concurrently without causing OOM
# issues. Use the `GPUS` and `PERCENT` options to control how 'much' of the GPUs you need for a test:
#
# GPUS 1 PERCENT 25 -> I need 25% of a single GPU GPUS 1 PERCENT 100 -> all of 1 GPU GPUS 2 PERCENT 200 -> all of 2 GPUs
# (will only run this test on 2 GPU machines)
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options)
  set(one_value NAME GPUS PERCENT)
  set(multi_value TARGETS CONFIGURATIONS)
  cmake_parse_arguments(_TRITON_FIL_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _TRITON_FIL_TEST_GPUS AND NOT DEFINED _TRITON_FIL_TEST_PERCENT)
    set(_TRITON_FIL_TEST_GPUS 1)
    set(_TRITON_FIL_TEST_PERCENT 15)
  endif()
  if(NOT DEFINED _TRITON_FIL_TEST_GPUS)
    set(_TRITON_FIL_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _TRITON_FIL_TEST_PERCENT)
    set(_TRITON_FIL_TEST_PERCENT 100)
  endif()

  find_library(
    TRITONSERVER_LIB
    tritonserver
    PATHS /home/phcho/Desktop/fil_backend
  )

  add_executable(${_TRITON_FIL_TEST_NAME} ${_TRITON_FIL_TEST_UNPARSED_ARGUMENTS})
  target_link_libraries(
    ${_TRITON_FIL_TEST_NAME}
    PRIVATE
      GTest::gtest
      GTest::gtest_main
      GTest::gmock
      $<$<BOOL:${TRITON_ENABLE_GPU}>:cuml>
      $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
      $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
      ${TREELITE_LIBS}
      nvforest::nvforest
      triton-core-serverstub
      triton-backend-utils
      "${TRITONSERVER_LIB}"
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      rapidjson)
  if(TRITON_ENABLE_GPU)
    target_compile_definitions(${_TRITON_FIL_TEST_NAME} PRIVATE -DTRITON_ENABLE_GPU=1)
  endif()

  target_compile_options(${_TRITON_FIL_TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${TRITON_FIL_CXX_FLAGS}>"
                                                        "$<$<COMPILE_LANGUAGE:CUDA>:${TRITON_FIL_CUDA_FLAGS}>")

  target_include_directories(${_TRITON_FIL_TEST_NAME} PRIVATE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>)
  set_target_properties(
    ${_TRITON_FIL_TEST_NAME}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON)
  if(TRITON_ENABLE_GPU)
    set_target_properties(${_TRITON_FIL_TEST_NAME} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  endif()

  set(_TRITON_FIL_TEST_COMPONENT_NAME testing)

  rapids_test_add(
    NAME ${_TRITON_FIL_TEST_NAME}
    COMMAND ${_TRITON_FIL_TEST_NAME}
    GPUS ${_TRITON_FIL_TEST_GPUS}
    PERCENT ${_TRITON_FIL_TEST_PERCENT}
    INSTALL_COMPONENT_SET ${_TRITON_FIL_TEST_COMPONENT_NAME})

endfunction()

ConfigureTest(NAME RAPIDS_TRITON_TEST
    rapids_triton/batch/batch.cpp
    rapids_triton/build_control.cpp
    rapids_triton/exceptions.cpp
    rapids_triton/memory/buffer.cpp
    rapids_triton/memory/detail/copy.cpp
    rapids_triton/memory/detail/owned_device_buffer.cpp
    rapids_triton/memory/resource.cpp
    rapids_triton/memory/types.cpp
    rapids_triton/tensor/dtype.cpp
    rapids_triton/tensor/tensor.cpp
    rapids_triton/test.cpp
    rapids_triton/triton/api/execute.cpp
    rapids_triton/triton/api/initialize.cpp
    rapids_triton/triton/api/instance_finalize.cpp
    rapids_triton/triton/api/instance_initialize.cpp
    rapids_triton/triton/api/model_finalize.cpp
    rapids_triton/triton/api/model_initialize.cpp
    rapids_triton/triton/backend.cpp
    rapids_triton/triton/config.cpp
    rapids_triton/triton/deployment.cpp
    rapids_triton/triton/device.cpp
    rapids_triton/triton/input.cpp
    rapids_triton/triton/logging.cpp
    rapids_triton/triton/model.cpp
    rapids_triton/triton/model_instance.cpp
    rapids_triton/triton/requests.cpp
    rapids_triton/triton/responses.cpp
    rapids_triton/triton/statistics.cpp
    rapids_triton/utils/const_agnostic.cpp
    rapids_triton/utils/narrow.cpp)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/rapids_triton_test)
