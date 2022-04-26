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

#include <cstdint>
#include <iostream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/ndarray.hpp>

namespace herring {

TEST(FilBackend, host_ndarray)
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
  auto arr1 = ndarray<int, 0, 1, 2>{data_layout1.data(), 2, 3, 4};

  auto arr2 = ndarray<int, 1, 0, 2>{data_layout2.data(), 2, 3, 4};

  ASSERT_EQ(arr1.data(), data_layout1.data());
  ASSERT_EQ(arr1.size(), data_layout1.size());
  ASSERT_EQ(arr1.axes(), 3);
  for (auto i = std::uint32_t{}; i < arr1.axes(); ++i) {
    ASSERT_EQ(arr1.dims()[i], i + 2);
  }

  for (auto i0 = std::uint32_t{}; i0 < 2; ++i0) {
    for (auto i1 = std::uint32_t{}; i1 < 3; ++i1) {
      for (auto i2 = std::uint32_t{}; i2 < 4; ++i2) {
        ASSERT_LT(arr1.get_index(i0, i1, i2), data_layout2.size());
        ASSERT_LT(arr2.get_index(i0, i1, i2), data_layout2.size());
        ASSERT_EQ(arr1.at(i0, i1, i2), arr2.at(i0, i1, i2));
        ASSERT_EQ(arr1.at(i0, i1, i2), arr1.other_at(data_layout1.data(), i0, i1, i2));
        ASSERT_EQ(arr2.at(i0, i1, i2), arr2.other_at(data_layout2.data(), i0, i1, i2));
      }
    }
  }
}

}
