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

#include <stdint.h>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kayak/buffer.hpp>
#include <kayak/device_type.hpp>
#include <kayak/flat_array.hpp>

namespace kayak {

__global__ void check_flat_array_access(
    bool* out,
    flat_array<array_encoding::dense, int> arr) {

  for (auto i = uint32_t{}; i < 3; ++i) {
    out[i] = (arr[i] == i + 1);
  }
}

TEST(FilBackend, dev_flat_array)
{
  auto data = std::vector<int>{1, 2, 3};
  auto buf = buffer<int>{
    buffer<int>{data.data(), data.size()},
    device_type::gpu
  };
  auto arr = flat_array<array_encoding::dense, int>{buf.data(), buf.size()};

  ASSERT_EQ(arr.size(), buf.size());
  ASSERT_EQ(arr.data(), buf.data());

  auto out_buf = buffer<bool>{data.size(), device_type::gpu};
  check_flat_array_access<<<1,1>>>(out_buf.data(), arr);
  auto out_buf_host = buffer<bool>{out_buf, device_type::cpu};
  cuda_check(cudaStreamSynchronize(0));
  for (auto i = std::uint32_t{}; i < data.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}

}
