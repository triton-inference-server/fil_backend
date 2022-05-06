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
#include <kayak/flat_array.hpp>
#include <kayak/structured_data.hpp>

namespace kayak {

TEST(FilBackend, structured_data_default)
{
  auto sd = structured_data<flat_array<array_encoding::dense, int>>{};
  EXPECT_EQ(sd.buffer().memory_type(), device_type::cpu);
  EXPECT_EQ(sd.buffer().size(), 0);
  EXPECT_EQ(sd.buffer().device_index(), 0);
  EXPECT_EQ(sd.obj().data(), nullptr);
  EXPECT_EQ(sd.obj().size(), 0);
}

TEST(FilBackend, make_structured_data)
{
  auto size = 7;
  auto mem_type = device_type::cpu;
  auto device = 0;
  auto sd = make_structured_data<flat_array<array_encoding::dense, int>, false>(
    size,
    mem_type,
    device,
    cuda_stream{},
    size
  );
  EXPECT_EQ(sd.buffer().memory_type(), device_type::cpu);
  EXPECT_EQ(sd.buffer().size(), size);
  EXPECT_EQ(sd.buffer().device_index(), device);
  EXPECT_NE(sd.buffer().data(), nullptr);
  EXPECT_EQ(sd.obj().data(), sd.buffer().data());
  EXPECT_EQ(sd.obj().size(), sd.buffer().size());
}

}
