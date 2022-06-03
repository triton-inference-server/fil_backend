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

#include <iostream>
#include <stdint.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>
#include <kayak/bitset.hpp>
#include <kayak/buffer.hpp>
#include <kayak/data_array.hpp>
#include <herring2/decision_forest.hpp>
#include <kayak/tree_layout.hpp>
#include <iterator>
#include <numeric>
#include <vector>

namespace herring {

TEST(FilBackend, host_forest_model)
{
  using forest_type = forest_model<float, kayak::tree_layout::depth_first, false, false, false, false>;

  auto offsets = std::vector<typename forest_type::offset_type>{
    6, 2, 0, 2, 0, 0, 0,
    4, 2, 0, 0, 2, 0, 2, 0, 0,
    2, 0, 0
  };
  auto offsets_buf = kayak::buffer(offsets.data(), offsets.size());
  auto cat1_data = typename forest_type::output_index_type{};
  auto categories1 = typename forest_type::category_set_type{&cat1_data};
  categories1.set(0);
  categories1.set(6);
  auto cat2_data = typename forest_type::output_index_type{};
  auto categories2 = typename forest_type::category_set_type{&cat2_data};
  categories2.set(4);

  auto values = std::vector<typename forest_type::node_value_type>{
    {.value = 1.0f},  // Tree: 0, Node 0
    {.value = 5.0f},  // Tree: 0, Node 1
    {.value = 6.0f},  // Tree: 0, Node 2
    {.value = 3.0f},  // Tree: 0, Node 3
    {.value = 4.0f},  // Tree: 0, Node 4
    {.value = 2.0f},  // Tree: 0, Node 5
    {.value = 0.0f},  // Tree: 0, Node 6
    {.value = 5.0f},  // Tree: 1, Node 0
    {.index = cat1_data},  // Tree: 1, Node 1, (Categories 0 and 6)
    {.value = 8.0f},  // Tree: 1, Node 2
    {.value = 7.0f},  // Tree: 1, Node 3
    {.index = cat2_data},  // Tree: 1, Node 4 (Category 4)
    {.value = 4.0f},  // Tree: 1, Node 5
    {.value = 2.0f},  // Tree: 1, Node 6
    {.value = 1.0f},  // Tree: 1, Node 7
    {.value = 0.0f},  // Tree: 1, Node 8
    {.value = 1.0f},  // Tree: 2, Node 0
    {.value = 0.0f},  // Tree: 2, Node 1
    {.value = 2.0f}  // Tree: 2, Node 2
  };

  auto values_buf = kayak::buffer(values.data(), values.size());
  auto features = std::vector<uint16_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto features_buf = kayak::buffer(features.data(), features.size());
  auto distant_vals = std::vector<bool>{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };
  auto distant_buf = kayak::buffer<bool>(std::begin(distant_vals), std::end(distant_vals));

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto tree_offsets_buf = kayak::buffer(tree_offsets.data(), tree_offsets.size());
  auto categorical_sizes = std::vector<uint32_t>{
    0, 0, 0, 0, 0, 0, 0,
    0, 8, 0, 0, 8, 0, 0, 0, 0,
    0, 0, 0
  };
  auto categorical_sizes_buf = kayak::buffer<uint32_t>(
    std::begin(categorical_sizes), std::end(categorical_sizes)
  );
  auto num_features = 2u;
  auto input_values = std::vector<float>{
    7.0f, 6.0f,
    0.0f, 1.0f,
    NAN, 1.0f,
  };
  auto target_rows = 129u;
  input_values.reserve(target_rows * num_features);
  for (auto i = std::size_t{}; i +6u < target_rows * num_features; ++i) {
    input_values.push_back(input_values[i % (num_features * 3)]);
  }

  auto input_values_buf = kayak::buffer(input_values.data(), input_values.size());
  auto input = kayak::data_array<kayak::data_layout::dense_row_major, float>{input_values_buf.data(), input_values.size() / num_features, num_features};

  auto test_forest = forest_type{
    1,
    input.cols(),
    std::move(tree_offsets_buf),
    std::move(values_buf),
    std::move(features_buf),
    std::move(offsets_buf),
    std::move(distant_buf),
    std::nullopt,
    std::move(categorical_sizes_buf),
    std::nullopt
  };

  auto output_values_buf = kayak::buffer<float>(input.rows());
  auto output = kayak::data_array<kayak::data_layout::dense_row_major, float>{output_values_buf.data(), input.rows(), 1};

  test_forest.predict(output, input);
  auto expected = std::vector<float>{13, 6, 4};

  for (auto i = std::size_t{}; i < output.rows(); ++i) {
    ASSERT_FLOAT_EQ(output.at(i, 0), expected[i % expected.size()]);
  }
}

}

