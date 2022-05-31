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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kayak/flat_array.hpp>

namespace kayak {

TEST(FilBackend, host_flat_array)
{
  auto data = std::vector<int>{1, 2, 3};
  auto arr = flat_array<array_encoding::dense, int>{data.data(), data.size()};

  ASSERT_EQ(arr.size(), data.size());
  ASSERT_EQ(arr.data(), data.data());

  for (auto i = std::uint32_t{}; i < data.size(); ++i) {
    ASSERT_EQ(arr.at(i), data[i]);
    ASSERT_EQ(arr[i], data[i]);
  }
}

TEST(FilBackend, host_make_flat_array)
{
  auto data = std::vector<int>{1, 2, 3};
  auto data_obj = make_flat_array<array_encoding::dense, int>(
    data.size()
  );
  auto& arr = data_obj.obj();
  auto& buf = data_obj.buffer();

  ASSERT_EQ(arr.size(), data.size());
  ASSERT_NE(buf.data(), data.data());
  ASSERT_EQ(buf.data(), arr.data());

  std::copy(std::begin(data), std::end(data), buf.data());

  for (auto i = std::uint32_t{}; i < data.size(); ++i) {
    ASSERT_EQ(arr.at(i), data[i]);
    ASSERT_EQ(arr[i], data[i]);
  }
}

}
