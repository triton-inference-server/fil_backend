#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# keep the files in alphabetical order!
add_executable(test_herring
    test/herring2/buffer.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/buffer.cu>
    test/herring2/cuda_check.cpp
    test/herring2/data_array.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/data_array.cu>
    test/herring2/device_id.cpp
    test/herring2/device_setter.cpp
    test/herring2/device_type.cpp
    test/herring2/gpu_support.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/gpu_support.cu>
    test/herring2/host_only_throw.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/host_only_throw.cu>
    test/herring2/index_type.cpp
    test/herring2/ndarray.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/ndarray.cu>
)

IF(TRITON_ENABLE_GPU)
  set_target_properties(test_herring
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             CUDA_STANDARD                       17
             CUDA_STANDARD_REQUIRED              ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(
    test_herring PUBLIC ENABLE_GPU
  )
else()
  set_target_properties(test_herring
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(test_herring
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
)

target_include_directories(test_herring
  PUBLIC  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/test>"
)

target_link_libraries(test_herring
PRIVATE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  gmock
  gmock_main
  GTest::gtest
  GTest::gtest_main
  $<TARGET_NAME_IF_EXISTS:conda_env>
)

