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
#include <cmath>
#include <kayak/bitset.hpp>
#include <kayak/buffer.hpp>
#include <kayak/data_array.hpp>
#include <herring2/forest.hpp>
#include <kayak/tree_layout.hpp>
#include <iterator>
#include <numeric>
#include <vector>

namespace herring {

TEST(FilBackend, small_host_forest)
{
  using forest_type = forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>;

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
  auto test_forest = forest_type{
    offsets_buf.size(),
    values_buf.data(),
    features_buf.data(),
    offsets_buf.data(),
    distant_buf.data(),
    tree_offsets_buf.size(),
    tree_offsets_buf.data(),
    uint32_t{1},
    nullptr,
    categorical_sizes_buf.data()
  };
  auto input_values = std::vector<float>{7.0f, 6.0f, 0.0f, 1.0f, NAN, 1.0f};
  auto input_values_buf = kayak::buffer(input_values.data(), input_values.size());
  auto input = kayak::data_array<kayak::data_layout::dense_row_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, false, false, true, false};
  auto missing_buf = kayak::buffer<bool>(std::begin(missing_vals), std::end(missing_vals));
  auto missing_input = kayak::data_array<kayak::data_layout::dense_row_major, bool>{
    missing_buf.data(), 3, 2
  };

  auto output = test_forest.evaluate_tree<true, true, false>(0, 0, input);
  ASSERT_FLOAT_EQ(output.at(0), 6.0f);
  output = test_forest.evaluate_tree<false, true, false>(0, 0, input);
  ASSERT_FLOAT_EQ(output.at(0), 6.0f);
  output = test_forest.evaluate_tree<true, true, false>(1, 0, input);
  ASSERT_FLOAT_EQ(output.at(0), 7.0f);
  output = test_forest.evaluate_tree<true, true, false>(2, 0, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, true, false>(2, 0, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);

  output = test_forest.evaluate_tree<true, true, false>(0, 1, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, true, false>(0, 1, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<true, true, false>(1, 1, input);
  ASSERT_FLOAT_EQ(output.at(0), 4.0f);
  output = test_forest.evaluate_tree<true, true, false>(2, 1, input);
  ASSERT_FLOAT_EQ(output.at(0), 2.0f);
  output = test_forest.evaluate_tree<false, true, false>(2, 1, input);
  ASSERT_FLOAT_EQ(output.at(0), 2.0f);

  output = test_forest.evaluate_tree<true, false>(0, 2, input, missing_input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<true, false, false>(0, 2, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, false>(0, 2, input, missing_input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, false, false>(0, 2, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<true, false>(1, 2, input, missing_input);
  ASSERT_FLOAT_EQ(output.at(0), 4.0f);
  output = test_forest.evaluate_tree<true, false, false>(1, 2, input);
  ASSERT_FLOAT_EQ(output.at(0), 4.0f);
  output = test_forest.evaluate_tree<true, false>(2, 2, input, missing_input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<true, false, false>(2, 2, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, false>(2, 2, input, missing_input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
  output = test_forest.evaluate_tree<false, false, false>(2, 2, input);
  ASSERT_FLOAT_EQ(output.at(0), 0.0f);
}

TEST(FilBackend, large_host_forest)
{
  using forest_type = forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint64_t, true>;

  auto offsets = std::vector<typename forest_type::offset_type>{
    6, 2, 0, 2, 0, 0, 0,
    4, 2, 0, 0, 2, 0, 2, 0, 0,
    2, 0, 0
  };
  auto offsets_buf = kayak::buffer(offsets.data(), offsets.size());

  auto categorical_sizes = std::vector<uint32_t>{
    0, 0, 0, 0, 0, 0, 0,
    0, 17, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0
  };

  auto categorical_sizes_buf = kayak::buffer<uint32_t>(
    std::begin(categorical_sizes), std::end(categorical_sizes)
  );

  auto cat_buf_size = std::accumulate(
    std::begin(categorical_sizes),
    std::end(categorical_sizes),
    uint32_t{},
    [](auto const& total, auto const& entry) {
      return total + (entry / sizeof(uint8_t) + (entry % sizeof(uint8_t) != 0));
    }
  );

  auto category_buf = kayak::buffer<uint8_t>(cat_buf_size);
  auto categories1_bins = categorical_sizes[8] / sizeof(uint8_t) + (categorical_sizes[8] % sizeof(uint8_t) != 0);
  auto categories1 = typename forest_type::category_set_type{
    category_buf.data(),
    categorical_sizes[8]
  };
  auto categories2 = typename forest_type::category_set_type{
    category_buf.data() + categories1_bins,
    categorical_sizes[11]
  };
  categories1.set(0);
  categories1.set(6);
  categories2.set(4);

  auto values = std::vector<typename forest_type::node_value_type>{
    {.value = 1.0f},  // Tree: 0, Node 0
    {.value = 5.0f},  // Tree: 0, Node 1
    {.value = 6.0f},  // Tree: 0, Node 2
    {.value = 3.0f},  // Tree: 0, Node 3
    {.index = 2},  // Tree: 0, Node 4
    {.index = 4},  // Tree: 0, Node 5
    {.index = 6},  // Tree: 0, Node 6
    {.value = 5.0f},  // Tree: 1, Node 0
    {.index = 0},  // Tree: 1, Node 1, (Categories 0 and 6)
    {.index = 8},  // Tree: 1, Node 2
    {.index = 10},  // Tree: 1, Node 3
    {.index = 1},  // Tree: 1, Node 4 (Category 4)
    {.index = 12},  // Tree: 1, Node 5
    {.value = 2.0f},  // Tree: 1, Node 6
    {.index = 14},  // Tree: 1, Node 7
    {.index = 16},  // Tree: 1, Node 8
    {.value = 1.0f},  // Tree: 2, Node 0
    {.index = 18},  // Tree: 2, Node 1
    {.index = 20}  // Tree: 2, Node 2
  };

  auto values_buf = kayak::buffer(values.data(), values.size());
  auto features = std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0,
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
  auto output_vals = std::vector<uint64_t>{
    6, 6, 4, 4, 2, 2, 0, 0, 8, 8, 7, 7, 4, 4, 1, 1, 0, 0, 0, 0, 2, 2
  };
  auto outputs_buf = kayak::buffer(output_vals.data(), output_vals.size());
  auto test_forest = forest_type{
    offsets_buf.size(),
    values_buf.data(),
    features_buf.data(),
    offsets_buf.data(),
    distant_buf.data(),
    tree_offsets_buf.size(),
    tree_offsets_buf.data(),
    uint32_t{2},
    outputs_buf.data(),
    categorical_sizes_buf.data(),
    category_buf.data()
  };
  auto input_values = std::vector<float>{7.0f, 0.0f, NAN, 6.0f, 1.0f, 1.0f};
  auto input_values_buf = kayak::buffer(input_values.data(), input_values.size());
  auto input = kayak::data_array<kayak::data_layout::dense_col_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, true, false, false, false};
  auto missing_buf = kayak::buffer<bool>(std::begin(missing_vals), std::end(missing_vals));
  auto missing_input = kayak::data_array<kayak::data_layout::dense_col_major, bool>{
    missing_buf.data(), 3, 2
  };

  auto output = test_forest.evaluate_tree<true, true, true>(0, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{6});
  ASSERT_EQ(output.at(1), uint64_t{6});
  output = test_forest.evaluate_tree<false, true, true>(0, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{6});
  ASSERT_EQ(output.at(1), uint64_t{6});
  output = test_forest.evaluate_tree<true, true, true>(1, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{7});
  ASSERT_EQ(output.at(1), uint64_t{7});
  output = test_forest.evaluate_tree<true, true, true>(2, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true, true>(2, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});

  output = test_forest.evaluate_tree<true, true, true>(0, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true, true>(0, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, true, true>(1, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{4});
  ASSERT_EQ(output.at(1), uint64_t{4});
  output = test_forest.evaluate_tree<true, true, true>(2, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{2});
  ASSERT_EQ(output.at(1), uint64_t{2});
  output = test_forest.evaluate_tree<false, true, true>(2, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{2});
  ASSERT_EQ(output.at(1), uint64_t{2});

  output = test_forest.evaluate_tree<true, true>(0, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, false, true>(0, 2, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(0, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, false, true>(0, 2, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, true>(1, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{4});
  ASSERT_EQ(output.at(1), uint64_t{4});
  output = test_forest.evaluate_tree<true, false, true>(1, 2, input);
  ASSERT_EQ(output.at(0), uint64_t{4});
  ASSERT_EQ(output.at(1), uint64_t{4});
  output = test_forest.evaluate_tree<true, true>(2, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, false, true>(2, 2, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(2, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, false, true>(2, 2, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
}

}
