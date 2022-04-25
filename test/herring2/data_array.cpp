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
#include <herring2/data_array.hpp>

namespace herring {

TEST(FilBackend, host_data_array)
{
  auto rows = uint32_t{2};
  auto cols = uint32_t{3};
  auto rm_data = std::vector<int>{1, 2, 3, 4, 5, 6};
  auto rm_arr = data_array<data_layout::dense_row_major, int>{rm_data.data(), rows, cols};
  auto cm_data = std::vector<int>{1, 4, 2, 5, 3, 6};
  auto cm_arr = data_array<data_layout::dense_col_major, int>{cm_data.data(), rows, cols};

  ASSERT_EQ(rm_arr.rows(), rows);
  ASSERT_EQ(rm_arr.cols(), cols);
  ASSERT_EQ(cm_arr.rows(), rows);
  ASSERT_EQ(cm_arr.cols(), cols);

  for (auto r = std::uint32_t{}; r < rows; ++r) {
    for (auto c = std::uint32_t{}; c < cols; ++c) {
      ASSERT_EQ(rm_arr.at(r, c), cm_arr.at(r, c));
      ASSERT_EQ(cm_arr.other_at(cm_arr.data(), r, c), rm_arr.at(r, c));
    }
  }
}

}
