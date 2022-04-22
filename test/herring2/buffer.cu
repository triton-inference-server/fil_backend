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

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/buffer.hpp>
#include <herring2/cuda_stream.hpp>
#include <herring2/device_type.hpp>

namespace herring {

__global__ void check_buffer_index(buffer<int> const& buf) {
  assert(buf[0] == 1);
  assert(buf[1] == 2);
  assert(buf[2] == 3);
  // TODO (wphicks): This is not working
}

TEST(FilBackend, device_buffer_index_operator)
{
  auto data = std::vector<int>{1, 2, 3};
  auto host_buf = buffer<int>(data.data(), data.size(), device_type::cpu);
  auto buf = buffer<int>(host_buf, device_type::gpu, 0, cuda_stream{});
  check_buffer_index<<<1,1>>>(buf);
  ASSERT_EQ(cudaStreamSynchronize(cuda_stream{}), cudaSuccess);
}

}
