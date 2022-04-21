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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <herring2/buffer.hpp>
#include <herring2/cuda_stream.hpp>
#include <herring2/device_type.hpp>
#include <herring2/exceptions.hpp>

namespace herring {

TEST(FilBackend, default_buffer)
{
  auto buf = buffer<int>();
  EXPECT_EQ(buf.memory_type(), device_type::cpu);
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.device_index(), 0);
}

TEST(FilBackend, device_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
  auto buf = buffer<int>(data.size(), device_type::gpu, 0, cuda_stream{});

  ASSERT_EQ(buf.memory_type(), device_type::gpu);
  ASSERT_EQ(buf.size(), data.size());
#ifdef TRITON_ENABLE_GPU
  ASSERT_NE(buf.data(), nullptr);

  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(buf.data()),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(buf.data()),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
#endif
}

TEST(FilBackend, non_owning_device_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
  auto* ptr_d = static_cast<int*>(nullptr);
#ifdef TRITON_ENABLE_GPU
  cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
  cudaMemcpy(static_cast<void*>(ptr_d),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
#endif
  auto buf = buffer<int>(ptr_d, data.size(), device_type::gpu);
#ifdef TRITON_ENABLE_GPU

  ASSERT_EQ(buf.memory_type(), device_type::gpu);
  ASSERT_EQ(buf.size(), data.size());
  ASSERT_EQ(buf.data(), ptr_d);

  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(buf.data()),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

  cudaFree(reinterpret_cast<void*>(ptr_d));
#endif
}

TEST(FilBackend, host_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buf = buffer<int>(data.size(), device_type::cpu, 0, 0);

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
  ASSERT_NE(buf.data(), nullptr);

  std::memcpy(
    static_cast<void*>(buf.data()), static_cast<void*>(data.data()), data.size() * sizeof(int));

  auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(FilBackend, non_owning_host_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buf = buffer<int>(data.data(), data.size(), device_type::cpu);

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
  ASSERT_EQ(buf.data(), data.data());

  auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(FilBackend, copy_buffer)
{
  auto data        = std::vector<int>{1, 2, 3};
  auto orig_buffer = buffer<int>(data.data(), data.size(), device_type::cpu);
  auto buf      = buffer<int>(orig_buffer);

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
  ASSERT_NE(buf.data(), orig_buffer.data());

  auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(FilBackend, move_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buf = buffer<int>(buffer<int>(data.data(), data.size(), device_type::cpu));

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
  ASSERT_EQ(buf.data(), data.data());

  auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(FilBackend, move_assignment_buffer)
{
  auto data = std::vector<int>{1, 2, 3};

#ifdef TRITON_ENABLE_GPU
  auto buf = buffer<int>{data.data(), data.size() - 1, device_type::gpu};
#else
  auto buf = buffer<int>{data.data(), data.size() - 1, device_type::cpu};
#endif
  buf      = buffer<int>{data.size(), device_type::cpu};

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
}

TEST(FilBackend, partial_buffer_copy)
{
  auto data1 = std::vector<int>{1, 2, 3, 4, 5};
  auto data2 = std::vector<int>{0, 0, 0, 0, 0};
  auto expected = std::vector<int>{0, 3, 4, 5, 0};
#ifdef TRITON_ENABLE_GPU
  auto buf1 = buffer<int>{data1.data(), data1.size(), device_type::gpu};
#else
  auto buf1 = buffer<int>{data1.data(), data1.size(), device_type::cpu};
#endif
  auto buf2 = buffer<int>{data2.data(), data2.size(), device_type::cpu};
  copy<true>(buf2, buf1, 1, 2, 3, cuda_stream{});
  copy<false>(buf2, buf1, 1, 2, 3, cuda_stream{});
  EXPECT_THROW(copy<true>(buf2, buf1, 1, 2, 4, cuda_stream{}), out_of_bounds);
}

}
