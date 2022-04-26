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
#include <herring2/buffer.hpp>
#include <herring2/data_array.hpp>
#include <herring2/device_type.hpp>

namespace herring {

__global__ void check_data_array_access(
    bool* out,
    data_array<data_layout::dense_row_major, int> rm_arr,
    data_array<data_layout::dense_col_major, int> cm_arr) {

  auto rows = uint32_t{2};
  auto cols = uint32_t{3};
  for (auto r = uint32_t{}; r < rows; ++r) {
    for (auto c = uint32_t{}; c < cols; ++c) {
      out[r * cols + c] = (rm_arr.at(r, c) == cm_arr.at(r, c));
    }
  }
}


TEST(FilBackend, dev_data_array)
{
  auto rows = uint32_t{2};
  auto cols = uint32_t{3};
  auto rm_data = std::vector<int>{1, 2, 3, 4, 5, 6};
  auto rm_buf = buffer<int>{
    buffer<int>{rm_data.data(), rm_data.size()},
    device_type::gpu
  };
  auto rm_arr = data_array<data_layout::dense_row_major, int>{rm_buf.data(), rows, cols};
  auto cm_data = std::vector<int>{1, 4, 2, 5, 3, 6};
  auto cm_buf = buffer<int>{
    buffer<int>{cm_data.data(), cm_data.size()},
    device_type::gpu
  };
  auto cm_arr = data_array<data_layout::dense_col_major, int>{cm_buf.data(), rows, cols};

  auto out_buf = buffer<bool>{rm_data.size(), device_type::gpu};

  check_data_array_access<<<1,1>>>(out_buf.data(), rm_arr, cm_arr);
  auto out_buf_host = buffer<bool>{out_buf, device_type::cpu};
  cuda_check(cudaStreamSynchronize(0));
  for (auto i = uint32_t{}; i < out_buf_host.size(); ++i) {
    ASSERT_EQ(out_buf_host.data()[i], true);
  }
}

}
