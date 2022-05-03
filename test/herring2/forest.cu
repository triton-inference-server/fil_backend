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
#include <herring2/bitset.hpp>
#include <herring2/buffer.hpp>
#include <herring2/data_array.hpp>
#include <herring2/device_type.hpp>
#include <herring2/forest.hpp>
#include <herring2/tree_layout.hpp>
#include <iterator>
#include <vector>

namespace herring {

using small_forest_type = forest<tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, bitset<32, uint32_t>>;

__global__ void check_small_forest(
    bool* out,
    small_forest_type test_forest,
    data_array<data_layout::dense_row_major, float> input) {

  auto output = test_forest.evaluate_tree<true, false>(0, 0, input);
  out[0] = (output.at(0) == 6.0f);
  output = test_forest.evaluate_tree<false, false>(0, 0, input);
  out[1] = (output.at(0) == 6.0f);
  output = test_forest.evaluate_tree<true, false>(1, 0, input);
  out[2] = (output.at(0) == 7.0f);
  output = test_forest.evaluate_tree<true, false>(2, 0, input);
  out[3] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<false, false>(2, 0, input);
  out[4] = (output.at(0) == 0.0f);

  output = test_forest.evaluate_tree<true, false>(0, 1, input);
  out[5] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<false, false>(0, 1, input);
  out[6] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<true, false>(1, 1, input);
  out[7] = (output.at(0) == 4.0f);
  output = test_forest.evaluate_tree<true, false>(2, 1, input);
  out[8] = (output.at(0) == 2.0f);
  output = test_forest.evaluate_tree<false, false>(2, 1, input);
  out[9] = (output.at(0) == 2.0f);

  output = test_forest.evaluate_tree<true, false>(0, 2, input, missing_input);
  out[10] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<false, false>(0, 2, input, missing_input);
  out[11] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<true, false>(1, 2, input, missing_input);
  out[12] = (output.at(0) == 4.0f);
  output = test_forest.evaluate_tree<true, false>(2, 2, input, missing_input);
  out[13] = (output.at(0) == 0.0f);
  output = test_forest.evaluate_tree<false, false>(2, 2, input, missing_input);
  out[14] = (output.at(0) == 0.0f);
}

TEST(FilBackend, small_dev_forest)
{
  using forest_type = small_forest_type;

  auto offsets = std::vector<typename forest_type::offset_type>{
    6, 2, 0, 2, 0, 0, 0,
    4, 2, 0, 0, 2, 0, 2, 0, 0,
    2, 0, 0
  };
  auto offsets_buf = buffer(
    std::begin(offsets), std::end(offsets), device_type::gpu
  );
  auto categories1 = typename forest_type::category_set_type{};
  categories1.set(0);
  categories1.set(6);
  auto categories2 = typename forest_type::category_set_type{};
  categories1.set(4);

  auto values = std::vector<typename forest_type::node_value_type>(offsets_buf.size());
  values[0].value = 1.0f;
  values[1].value = 5.0f;
  values[2].value = 6.0f;
  values[3].value = 3.0f;
  values[4].value = 4.0f;
  values[5].value = 2.0f;
  values[6].value = 0.0f;
  values[7 + 0].value = 5.0f;
  values[7 + 1].index = categories1;
  values[7 + 2].value = 8.0f;
  values[7 + 3].value = 7.0f;
  values[7 + 4].index = categories2;
  values[7 + 5].value = 4.0f;
  values[7 + 6].value = 2.0f;
  values[7 + 7].value = 1.0f;
  values[7 + 8].value = 0.0f;
  values[16 + 0].value = 1.0f;
  values[16 + 1].value = 0.0f;
  values[16 + 2].value = 2.0f;

  auto values_buf = buffer(
    std::begin(values), std::end(values), device_type::gpu
  );
  auto features = std::vector<uint16_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto features_buf = buffer(
    std::begin(features), std::end(features), device_type::gpu
  );
  auto distant_vals = std::vector<bool>{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };
  auto distant_buf = buffer<bool>(
    std::begin(distant_vals), std::end(distant_vals),  device_type::gpu
  );

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto tree_offsets_buf = buffer(
    std::begin(tree_offsets),
    std::end(tree_offsets),
    device_type::gpu
  );
  auto categorical_nodes = std::vector<bool>{
    false, false, false, false, false, false, false,
    false, true, false, false, true, false, false, false, false,
    false, false, false
  };
  auto categorical_nodes_buf = buffer<bool>(
    std::begin(categorical_nodes), std::end(categorical_nodes),
    device_type::gpu
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
    categorical_nodes_buf.data()
  };
  auto input_values = std::vector<float>{7.0f, 6.0f, 0.0f, 1.0f, NAN, 1.0f};
  auto input_values_buf = buffer(
    std::begin(input_values), std::end(input_values), device_type::gpu
  );
  auto input = data_array<data_layout::dense_row_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, false, false, true, false};
  auto missing_buf = buffer<bool>(
    std::begin(missing_vals), std::end(missing_vals), device_type::gpu
  );
  auto missing_input = data_array<data_layout::dense_row_major, bool>{
    missing_buf.data(), 3, 2
  };
}

TEST(FilBackend, large_host_forest)
{
  using forest_type = forest<tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint64_t, bitset<1024, uint8_t>>;

  auto offsets = std::vector<typename forest_type::offset_type>{
    6, 2, 0, 2, 0, 0, 0,
    4, 2, 0, 0, 2, 0, 2, 0, 0,
    2, 0, 0
  };
  auto offsets_buf = buffer(offsets.data(), offsets.size());
  auto categories = std::vector<typename forest_type::category_set_type>(2);
  categories[0].set(0);
  categories[0].set(6);
  categories[1].set(4);
  auto categories_buf = buffer(categories.data(), categories.size());

  auto values = std::vector<typename forest_type::node_value_type>(offsets_buf.size());
  values[0].value = 1.0;
  values[1].value = 5.0;
  values[2].index = 0;
  values[3].value = 3.0;
  values[4].index = 2;
  values[5].index = 4;
  values[6].index = 6;
  values[7 + 0].value = 5.0;
  values[7 + 1].index = 0;
  values[7 + 2].index = 8;
  values[7 + 3].index = 10;
  values[7 + 4].index = 1;
  values[7 + 5].index = 12;
  values[7 + 6].value = 2.0;
  values[7 + 7].index = 14;
  values[7 + 8].index = 16;
  values[16 + 0].value = 1.0;
  values[16 + 1].index = 18;
  values[16 + 2].index = 20;

  auto values_buf = buffer(values.data(), values.size());
  auto features = std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto features_buf = buffer(features.data(), features.size());
  auto distant_vals = std::vector<bool>{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };
  auto distant_buf = buffer<bool>(std::begin(distant_vals), std::end(distant_vals));

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto tree_offsets_buf = buffer(tree_offsets.data(), tree_offsets.size());
  auto categorical_nodes = std::vector<bool>{
    false, false, false, false, false, false, false,
    false, true, false, false, true, false, false, false, false,
    false, false, false
  };
  auto categorical_nodes_buf = buffer<bool>(
    std::begin(categorical_nodes), std::end(categorical_nodes)
  );
  auto output_vals = std::vector<uint64_t>{
    6, 6, 4, 4, 2, 2, 0, 0, 8, 8, 7, 7, 4, 4, 1, 1, 0, 0, 0, 0, 2, 2
  };
  auto outputs_buf = buffer(output_vals.data(), output_vals.size());
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
    categorical_nodes_buf.data(),
    categories_buf.data()
  };
  auto input_values = std::vector<float>{7.0f, 6.0f, 0.0f, 1.0f, NAN, 1.0f};
  auto input_values_buf = buffer(input_values.data(), input_values.size());
  auto input = data_array<data_layout::dense_row_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, false, false, true, false};
  auto missing_buf = buffer<bool>(std::begin(missing_vals), std::end(missing_vals));
  auto missing_input = data_array<data_layout::dense_row_major, bool>{
    missing_buf.data(), 3, 2
  };

  auto output = test_forest.evaluate_tree<true, true>(0, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{6});
  ASSERT_EQ(output.at(1), uint64_t{6});
  output = test_forest.evaluate_tree<false, true>(0, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{6});
  ASSERT_EQ(output.at(1), uint64_t{6});
  output = test_forest.evaluate_tree<true, true>(1, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{7});
  ASSERT_EQ(output.at(1), uint64_t{7});
  output = test_forest.evaluate_tree<true, true>(2, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(2, 0, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});

  output = test_forest.evaluate_tree<true, true>(0, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(0, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, true>(1, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{4});
  ASSERT_EQ(output.at(1), uint64_t{4});
  output = test_forest.evaluate_tree<true, true>(2, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{2});
  ASSERT_EQ(output.at(1), uint64_t{2});
  output = test_forest.evaluate_tree<false, true>(2, 1, input);
  ASSERT_EQ(output.at(0), uint64_t{2});
  ASSERT_EQ(output.at(1), uint64_t{2});

  output = test_forest.evaluate_tree<true, true>(0, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(0, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<true, true>(1, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{4});
  ASSERT_EQ(output.at(1), uint64_t{4});
  output = test_forest.evaluate_tree<true, true>(2, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
  output = test_forest.evaluate_tree<false, true>(2, 2, input, missing_input);
  ASSERT_EQ(output.at(0), uint64_t{0});
  ASSERT_EQ(output.at(1), uint64_t{0});
}

}
