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
#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/buffer.hpp>
#include <herring2/cuda_stream.hpp>
#include <herring2/device_type.hpp>

namespace herring {

__global__ void check_buffer_index(int* buf) {
  printf("Greetings!\n");
  if (buf[0] == 1) {
    buf[0] = 4;
    printf("Looking good!\n");
    if (buf[0] == 4) {
      printf("Why thank you!\n");
    }
  }
  if (buf[1] == 2) {
    buf[1] = 5;
    printf("Looking great!\n");
  }
  if (buf[2] == 3) {
    buf[2] = 6;
    printf("Looking amazing!\n");
  }
}

TEST(FilBackend, device_buffer_index_operator)
{
  auto data = std::vector<int>{1, 2, 3};
  auto expected = std::vector<int>{4, 5, 6};
  auto buf = buffer<int>(
    buffer<int>(data.data(), data.size(), device_type::cpu),
    device_type::gpu,
    0,
    cuda_stream{}
  );
  check_buffer_index<<<1,1>>>(buf.data());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaStreamSynchronize(cuda_stream{}), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto data_out = std::vector<int>(expected.size());
  cudaDeviceSynchronize();
  auto host_buf = buffer<int>(data_out.data(), data_out.size(), device_type::cpu);
  copy<true>(host_buf, buf);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaStreamSynchronize(cuda_stream{}), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  // copy<true>(host_buf, buf);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
}

}