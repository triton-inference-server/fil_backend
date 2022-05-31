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
#include <gtest/gtest.h>
#include <kayak/bitset.hpp>
#include <kayak/detail/index_type.hpp>

namespace kayak {
TEST(FilBackend, default_bitset) {
  auto constexpr small_size = raw_index_t{8};
  auto constexpr large_size = raw_index_t{33};
  using small_set_t = bitset<uint8_t>;
  using large_set_t = bitset<uint32_t>;
  auto small_data = uint8_t{};
  auto small_set = small_set_t{&small_data};
  auto large_data = std::vector<uint32_t>(2);
  auto large_set = large_set_t{large_data.data(), large_size};

  ASSERT_EQ(small_set.size(), small_size);
  ASSERT_EQ(large_set.size(), large_size);
  ASSERT_EQ(small_set.bin_width, 8);
  ASSERT_EQ(large_set.bin_width, 32);
  ASSERT_EQ(small_set.bin_count(), 1);
  ASSERT_EQ(large_set.bin_count(), 2);

  for (auto i = raw_index_t{}; i < small_size; ++i) {
    ASSERT_EQ(small_set.test(i), false);
  }
  for (auto i = raw_index_t{}; i < large_size; ++i) {
    ASSERT_EQ(large_set.test(i), false);
  }

  small_set.flip();
  large_set.flip();

  for (auto i = raw_index_t{}; i < small_size; ++i) {
    ASSERT_EQ(small_set.test(i), true);
    small_set.clear(i);
    ASSERT_EQ(small_set.test(i), false);
    large_set.set(i);
    ASSERT_EQ(large_set.test(i), true);
  }
  for (auto i = raw_index_t{}; i < large_size; ++i) {
    ASSERT_EQ(large_set.test(i), true);
    large_set.clear(i);
    ASSERT_EQ(large_set.test(i), false);
    large_set.set(i);
    ASSERT_EQ(large_set.test(i), true);
  }
}

}
