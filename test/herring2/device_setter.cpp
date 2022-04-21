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

#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#endif
#include <gtest/gtest.h>
#include <herring2/cuda_check.hpp>
#include <herring2/device_setter.hpp>

namespace herring {
TEST(FilBackend, device_setter) {
#ifdef ENABLE_GPU
  auto dev_count = int{};
  cuda_check(cudaGetDeviceCount(&dev_count));
  auto init_dev = int{};
  cuda_check(cudaSetDevice(init_dev));
  cuda_check(cudaGetDevice(&init_dev));

  if (dev_count > 0) {
    auto target_dev = dev_count - 1;
    {
      auto setter = device_setter{target_dev};
      auto new_dev = int{};
      cuda_check(cudaGetDevice(&new_dev));
      ASSERT_EQ(new_dev, target_dev);
    }
    auto final_dev = int{};
    cuda_check(cudaGetDevice(&final_dev));
    ASSERT_EQ(final_dev, init_dev);
  }
#else
  auto setter = device_setter{0};
#endif
}
}
