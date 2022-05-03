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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <array>
#include <herring2/bitset.hpp>
#include <herring2/buffer.hpp>
#include <herring2/forest.hpp>
#include <herring2/tree_layout.hpp>
#include <vector>

namespace herring {

TEST(FilBackend, host_forest)
{
  using forest_type = forest<tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, bitset<32, uint32_t>>;

  auto offsets = std::vector<typename forest_type::offset_type>{6, 2, 0, 2, 0, 0, 0,
                                       4, 2, 0, 0, 2, 0, 2, 0, 0,
                                       2, 0, 0};
  auto categories1 = typename forest_type::category_set_type{};
  categories1.set(0);
  categories1.set(2);
  auto categories2 = typename forest_type::category_set_type{};
  categories1.set(1);

  auto values = std::vector<typename forest_type::node_value_type>(offsets.size());
  values[0].value = 0.0f;
  values[1].value = 0.1f;
  values[2].value = 0.2f;
  values[3].value = 0.3f;
  values[4].value = 0.4f;
  values[5].value = 0.5f;
  values[6].value = 0.6f;
  values[7 + 0].value = 0.0f;
  values[7 + 1].categories = categories1;
  values[7 + 2].value = 0.2f;
  values[7 + 3].value = 0.3f;
  values[7 + 4].categories = categories2;
  values[7 + 5].value = 0.5f;
  values[7 + 6].value = 0.6f;
  values[7 + 7].value = 0.7f;
  values[7 + 8].value = 0.8f;
  values[16 + 0].value = 0.0f;
  values[16 + 1].value = 0.1f;
  values[16 + 2].value = 0.2f;
  auto features = std::vector<uint16_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto distant_vals = std::array{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto categorical_nodes = std::array{
    false, false, false, false, false, false, false,
    false, true, false, false, true, false, false, false, false,
    false, false, false
  };
  auto test_forest = forest_type{
    offsets.size(),
    values.data(),
    features.data(),
    offsets.data(),
    static_cast<bool*>(distant_vals.data()),
    tree_offsets.size(),
    tree_offsets.data(),
    uint32_t{1},
    nullptr,
    static_cast<bool*>(categorical_nodes.data())
  };
}

}
