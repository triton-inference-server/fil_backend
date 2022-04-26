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
#include <iostream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/buffer.hpp>
#include <herring2/device_type.hpp>
#include <herring2/ndarray.hpp>

namespace herring {

__global__ void check_ndarray_access(
    bool* out,
    ndarray<int, 0, 1, 2> arr1,
    ndarray<int, 1, 0, 2> arr2) {

  for (auto i0 = uint32_t{}; i0 < 2; ++i0) {
    for (auto i1 = uint32_t{}; i1 < 3; ++i1) {
      for (auto i2 = uint32_t{}; i2 < 4; ++i2) {
        out[i0 * 12 + i1 * 4 + i2] = (
          arr1.get_index(i0, i1, i2) < arr1.size() &&
          arr2.get_index(i0, i1, i2) < arr2.size() &&
          arr1.at(i0, i1, i2) == arr2.at(i0, i1, i2) &&
          arr1.at(i0, i1, i2) == arr1.other_at(arr1.data(), i0, i1, i2) &&
          arr2.at(i0, i1, i2) == arr2.other_at(arr2.data(), i0, i1, i2)
        );
      }
    }
  }
}

TEST(FilBackend, dev_ndarray)
{
  auto data_layout1 = std::vector<int>{
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,

    12, 13, 14, 15,
    16, 17, 18, 19,
    20, 21, 22, 23
  };
  auto data_layout2 = std::vector<int>{
     0,  1,  2,  3,  12, 13, 14, 15,
     4,  5,  6,  7,  16, 17, 18, 19,
     8,  9, 10, 11,  20, 21, 22, 23
  };
  auto buf1 = buffer<int>{
    buffer<int>{data_layout1.data(), data_layout1.size()},
    device_type::gpu
  };
  auto arr1 = ndarray<int, 0, 1, 2>{buf1.data(), 2, 3, 4};

  auto buf2 = buffer<int>{
    buffer<int>{data_layout2.data(), data_layout2.size()},
    device_type::gpu
  };
  auto arr2 = ndarray<int, 1, 0, 2>{buf2.data(), 2, 3, 4};

  ASSERT_EQ(arr1.data(), buf1.data());
  ASSERT_EQ(arr1.size(), buf1.size());
  ASSERT_EQ(arr1.axes(), 3);
  for (auto i = std::uint32_t{}; i < arr1.axes(); ++i) {
    ASSERT_EQ(arr1.dims()[i], i + 2);
  }

  auto out_buf = buffer<bool>{data_layout1.size(), device_type::gpu};
  check_ndarray_access<<<1,1>>>(out_buf.data(), arr1, arr2);
  auto out_buf_host = buffer<bool>{out_buf, device_type::cpu};
  cuda_check(cudaStreamSynchronize(0));
  for (auto i = uint32_t{}; i < out_buf_host.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}

}
