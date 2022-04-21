/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <herring2/gpu_support.hpp>

namespace herring {

HOST void host_func() {
  printf("HOST macro working\n");
}
DEVICE void device_func() {
  printf("DEVICE macro working\n");
}
GLOBAL void global_func() {
  device_func();
  printf("GLOBAL macro working\n");
}
TEST(FilBackend, gpu_macros) {
  host_func();
  global_func<<<1,1>>>();
  auto err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess);
}

}
