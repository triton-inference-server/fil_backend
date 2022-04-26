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
#include <herring2/tree.hpp>

namespace herring {

TEST(FilBackend, host_tree)
{
  auto df_out = std::vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto bf_out = std::vector<int>{0, 1, 6, 2, 3, 4, 5};

  auto df_data = std::vector<int>{6, 2, 0, 2, 0, 0, 0};
  auto df_tree = tree<tree_layout::depth_first, int>{df_data.data(), df_data.size()};
  auto bf_data = std::vector<int>{2, 3, 0, 0, 2, 0, 0};
  auto bf_tree = tree<tree_layout::breadth_first, int>{bf_data.data(), bf_data.size()};

  ASSERT_EQ(df_tree.data(), df_data.data());
  ASSERT_EQ(bf_tree.data(), bf_data.data());
  ASSERT_EQ(df_tree.size(), df_data.size());
  ASSERT_EQ(bf_tree.size(), bf_data.size());

  auto paths = std::vector<std::vector<bool>>{
    {},
    {false},
    {false, false},
    {false, true},
    {false, true, false},
    {false, true, true},
    {true},
  };
  for (auto i = std::uint32_t{}; i < paths.size(); ++i) {
    auto const& path = paths[i];
    auto df_index = std::uint32_t{};
    auto bf_index = std::uint32_t{};
    for (auto cond : path) {
      df_index += df_tree.next_offset(df_index, cond);
      bf_index += bf_tree.next_offset(bf_index, cond);
    }
    ASSERT_EQ(df_out[df_index], i);
    ASSERT_EQ(df_out[df_index], i);
  }
}

}
