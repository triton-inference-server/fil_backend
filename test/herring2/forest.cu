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
#include <herring2/forest.hpp>
#include <kayak/bitset.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/data_array.hpp>
#include <kayak/device_type.hpp>
#include <kayak/tree_layout.hpp>
#include <iterator>
#include <numeric>
#include <vector>

namespace herring {

using small_forest_type = forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>;
using large_forest_type = forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint64_t, true>;

/* __global__ void check_small_forest(
    bool* out,
    small_forest_type test_forest,
    kayak::data_array<kayak::data_layout::dense_row_major, float> input,
    kayak::data_array<kayak::data_layout::dense_row_major, bool> missing_input
  ) {

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
  auto offsets_buf = kayak::buffer<typename forest_type::offset_type>(
    std::begin(offsets), std::end(offsets), kayak::device_type::gpu
  );
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

  auto values_buf = kayak::buffer<typename forest_type::node_value_type>(
    std::begin(values), std::end(values), kayak::device_type::gpu
  );
  auto features = std::vector<uint16_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto features_buf = kayak::buffer<uint16_t>(
    std::begin(features), std::end(features), kayak::device_type::gpu
  );
  auto distant_vals = std::vector<bool>{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };
  auto distant_buf = kayak::buffer<bool>(
    std::begin(distant_vals), std::end(distant_vals),  kayak::device_type::gpu
  );

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto tree_offsets_buf = kayak::buffer<uint32_t>(
    std::begin(tree_offsets),
    std::end(tree_offsets),
    kayak::device_type::gpu
  );
  auto categorical_sizes = std::vector<uint32_t>{
    0, 0, 0, 0, 0, 0, 0,
    0, 8, 0, 0, 8, 0, 0, 0, 0,
    0, 0, 0
  };
  auto categorical_sizes_buf = kayak::buffer<uint32_t>(
    std::begin(categorical_sizes), std::end(categorical_sizes),
    kayak::device_type::gpu
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
  auto input_values_buf = kayak::buffer<float>(
    std::begin(input_values), std::end(input_values), kayak::device_type::gpu
  );
  auto input = kayak::data_array<kayak::data_layout::dense_row_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, false, false, true, false};
  auto missing_buf = kayak::buffer<bool>(
    std::begin(missing_vals), std::end(missing_vals), kayak::device_type::gpu
  );
  auto missing_input = kayak::data_array<kayak::data_layout::dense_row_major, bool>{
    missing_buf.data(), 3, 2
  };
  auto out_buf = kayak::buffer<bool>{15, kayak::device_type::gpu};
  check_small_forest<<<1,1>>>(
    out_buf.data(), test_forest, input, missing_input
  );
  auto out_buf_host = kayak::buffer<bool>{out_buf, kayak::device_type::cpu};
  kayak::cuda_check(cudaStreamSynchronize(0));
  for (auto i = uint32_t{}; i < out_buf_host.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}

__global__ void check_large_forest(
    bool* out,
    large_forest_type test_forest,
    kayak::data_array<kayak::data_layout::dense_row_major, float> input,
    kayak::data_array<kayak::data_layout::dense_row_major, bool> missing_input
  ) {

  auto output = test_forest.evaluate_tree<true, true>(0, 0, input);
  out[0] = (output.at(0) == 6.0f);
  out[1] = (output.at(1) == 6.0f);
  output = test_forest.evaluate_tree<false, true>(0, 0, input);
  out[2] = (output.at(0) == 6.0f);
  out[3] = (output.at(1) == 6.0f);
  output = test_forest.evaluate_tree<true, true>(1, 0, input);
  out[4] = (output.at(0) == 7.0f);
  out[5] = (output.at(1) == 7.0f);
  output = test_forest.evaluate_tree<true, true>(2, 0, input);
  out[6] = (output.at(0) == 0.0f);
  out[7] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<false, true>(2, 0, input);
  out[8] = (output.at(0) == 0.0f);
  out[9] = (output.at(1) == 0.0f);

  output = test_forest.evaluate_tree<true, true>(0, 1, input);
  out[10] = (output.at(0) == 0.0f);
  out[11] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<false, true>(0, 1, input);
  out[12] = (output.at(0) == 0.0f);
  out[13] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<true, true>(1, 1, input);
  out[14] = (output.at(0) == 4.0f);
  out[15] = (output.at(1) == 4.0f);
  output = test_forest.evaluate_tree<true, true>(2, 1, input);
  out[16] = (output.at(0) == 2.0f);
  out[17] = (output.at(1) == 2.0f);
  output = test_forest.evaluate_tree<false, true>(2, 1, input);
  out[18] = (output.at(0) == 2.0f);
  out[19] = (output.at(1) == 2.0f);

  output = test_forest.evaluate_tree<true, true>(0, 2, input, missing_input);
  out[20] = (output.at(0) == 0.0f);
  out[21] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<false, true>(0, 2, input, missing_input);
  out[22] = (output.at(0) == 0.0f);
  out[23] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<true, true>(1, 2, input, missing_input);
  out[24] = (output.at(0) == 4.0f);
  out[25] = (output.at(1) == 4.0f);
  output = test_forest.evaluate_tree<true, true>(2, 2, input, missing_input);
  out[26] = (output.at(0) == 0.0f);
  out[27] = (output.at(1) == 0.0f);
  output = test_forest.evaluate_tree<false, true>(2, 2, input, missing_input);
  out[28] = (output.at(0) == 0.0f);
  out[29] = (output.at(1) == 0.0f);
}

TEST(FilBackend, large_dev_forest)
{
  using forest_type = large_forest_type;

  auto offsets = std::vector<typename forest_type::offset_type>{
    6, 2, 0, 2, 0, 0, 0,
    4, 2, 0, 0, 2, 0, 2, 0, 0,
    2, 0, 0
  };
  auto offsets_buf = kayak::buffer<typename forest_type::offset_type>(
    std::begin(offsets), std::end(offsets), kayak::device_type::gpu
  );

  auto categorical_sizes = std::vector<uint32_t>{
    0, 0, 0, 0, 0, 0, 0,
    0, 17, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0
  };

  auto cat_buf_size = std::accumulate(
    std::begin(categorical_sizes),
    std::end(categorical_sizes),
    uint32_t{},
    [](auto const& total, auto const& entry) {
      return total + (entry / sizeof(uint8_t) + (entry % sizeof(uint8_t) != 0));
    }
  );

  auto categorical_data = std::vector<uint8_t>(cat_buf_size);

  auto categories1_bins = categorical_sizes[8] / sizeof(uint8_t) + (categorical_sizes[8] % sizeof(uint8_t) != 0);

  auto categories1 = typename forest_type::category_set_type{
    categorical_data.data(),
    categorical_sizes[8]
  };
  auto categories2 = typename forest_type::category_set_type{
    categorical_data.data() + categories1_bins,
    categorical_sizes[11]
  };
  categories1.set(0);
  categories1.set(6);
  categories2.set(4);

  auto categorical_sizes_buf = kayak::buffer<uint32_t>(
    std::begin(categorical_sizes),
    std::end(categorical_sizes),
    kayak::device_type::gpu
  );
  auto category_buf = kayak::buffer<uint8_t>(
    std::begin(categorical_data),
    std::end(categorical_data),
    kayak::device_type::gpu
  );

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

  auto values_buf = kayak::buffer<typename forest_type::node_value_type>(
    std::begin(values), std::end(values), kayak::device_type::gpu
  );
  auto features = std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0,
                                        0, 1, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0};
  auto features_buf = kayak::buffer<uint32_t>(
    std::begin(features), std::end(features), kayak::device_type::gpu
  );
  auto distant_vals = std::vector<bool>{
    true, false, false, false, false, false, false,
    true, false, false, false, false, false, false, false, false,
    false, false, false
  };
  auto distant_buf = kayak::buffer<bool>(
    std::begin(distant_vals), std::end(distant_vals),  kayak::device_type::gpu
  );

  auto tree_offsets = std::vector<uint32_t>{0, 7, 16};
  auto tree_offsets_buf = kayak::buffer<uint32_t>(
    std::begin(tree_offsets),
    std::end(tree_offsets),
    kayak::device_type::gpu
  );
  auto output_vals = std::vector<uint64_t>{
    6, 6, 4, 4, 2, 2, 0, 0, 8, 8, 7, 7, 4, 4, 1, 1, 0, 0, 0, 0, 2, 2
  };
  auto outputs_buf = kayak::buffer<uint64_t>(
    std::begin(output_vals),
    std::end(output_vals),
    kayak::device_type::gpu
  );
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
  auto input_values = std::vector<float>{7.0f, 6.0f, 0.0f, 1.0f, NAN, 1.0f};
  auto input_values_buf = kayak::buffer<float>(
    std::begin(input_values), std::end(input_values), kayak::device_type::gpu
  );
  auto input = kayak::data_array<kayak::data_layout::dense_row_major, float>{input_values_buf.data(), 3, 2};
  auto missing_vals = std::vector<bool>{false, false, false, false, true, false};
  auto missing_buf = kayak::buffer<bool>(
    std::begin(missing_vals), std::end(missing_vals), kayak::device_type::gpu
  );
  auto missing_input = kayak::data_array<kayak::data_layout::dense_row_major, bool>{
    missing_buf.data(), 3, 2
  };
  auto out_buf = kayak::buffer<bool>{30, kayak::device_type::gpu};
  check_large_forest<<<1,1>>>(
    out_buf.data(), test_forest, input, missing_input
  );
  auto out_buf_host = kayak::buffer<bool>{out_buf, kayak::device_type::cpu};
  kayak::cuda_check(cudaStreamSynchronize(0));
  for (auto i = uint32_t{}; i < out_buf_host.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}*/

}
