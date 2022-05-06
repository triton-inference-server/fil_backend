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
add_executable(test_kayak
    test/kayak/bitset.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/bitset.cu>
    test/kayak/buffer.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/buffer.cu>
    test/kayak/cuda_check.cpp
    test/kayak/data_array.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/data_array.cu>
    test/kayak/device_id.cpp
    test/kayak/device_setter.cpp
    test/kayak/device_type.cpp
    test/kayak/flat_array.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/flat_array.cu>
    test/kayak/gpu_support.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/gpu_support.cu>
    test/kayak/host_only_throw.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/host_only_throw.cu>
    test/kayak/index_type.cpp
    test/kayak/ndarray.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/ndarray.cu>
    test/kayak/tree.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/kayak/tree.cu>
)

add_executable(test_herring
    test/herring2/forest.cpp
    $<$<BOOL:${TRITON_ENABLE_GPU}>:test/herring2/forest.cu>
)

IF(TRITON_ENABLE_GPU)
  set_target_properties(test_kayak
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
    test_kayak PUBLIC ENABLE_GPU
  )
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
  set_target_properties(test_kayak
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  set_target_properties(test_herring
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(test_kayak
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
)
target_compile_options(test_herring
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
)

target_include_directories(test_kayak
  PUBLIC  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/test>"
)
target_include_directories(test_herring
  PUBLIC  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/test>"
)

target_link_libraries(test_kayak
PRIVATE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  gmock
  gmock_main
  GTest::gtest
  GTest::gtest_main
  $<TARGET_NAME_IF_EXISTS:conda_env>
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

