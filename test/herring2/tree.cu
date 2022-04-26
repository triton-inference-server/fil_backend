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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/buffer.hpp>
#include <herring2/cuda_check.hpp>
#include <herring2/device_type.hpp>
#include <herring2/tree.hpp>

namespace herring {

__global__ void check_tree_traversal(
    bool* out,
    tree<tree_layout::depth_first, int> df_tree,
    tree<tree_layout::breadth_first, int> bf_tree) {

  int df_out[] = {0, 1, 2, 3, 4, 5, 6};
  int bf_out[] = {0, 1, 6, 2, 3, 4, 5};
  bool path[] = {false, true, false};
  auto df_index = uint32_t{};
  auto bf_index = uint32_t{};
  for (auto cond : path) {
    df_index += df_tree.next_offset(df_index, cond);
    bf_index += bf_tree.next_offset(bf_index, cond);
  }
  out[0] = df_out[df_index] == 4;
  out[1] = bf_out[bf_index] == 4;
}

TEST(FilBackend, dev_tree)
{
  auto df_out = std::vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto bf_out = std::vector<int>{0, 1, 6, 2, 3, 4, 5};

  auto df_data = std::vector<int>{6, 2, 0, 2, 0, 0, 0};
  auto df_buf = buffer<int>{
    buffer<int>{df_data.data(), df_data.size()},
    device_type::gpu
  };
  auto df_tree = tree<tree_layout::depth_first, int>{df_buf.data(), df_buf.size()};
  auto bf_data = std::vector<int>{2, 3, 0, 0, 2, 0, 0};
  auto bf_buf = buffer<int>{
    buffer<int>{bf_data.data(), bf_data.size()},
    device_type::gpu
  };
  auto bf_tree = tree<tree_layout::breadth_first, int>{bf_buf.data(), bf_buf.size()};

  ASSERT_EQ(df_tree.data(), df_buf.data());
  ASSERT_EQ(bf_tree.data(), bf_buf.data());
  ASSERT_EQ(df_tree.size(), df_buf.size());
  ASSERT_EQ(bf_tree.size(), bf_buf.size());

  auto out_buf = buffer<bool>{2, device_type::gpu};
  check_tree_traversal<<<1,1>>>(out_buf.data(), df_tree, bf_tree);
  auto out_buf_host = buffer<bool>{out_buf, device_type::cpu};
  cuda_check(cudaStreamSynchronize(0));
  for (auto i = uint32_t{}; i < out_buf_host.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}

}
