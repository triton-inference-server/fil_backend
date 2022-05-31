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

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <kayak/detail/index_type.hpp>

namespace kayak {
TEST(FilBackend, index_type) {
  auto zero = 0;
  auto one = 1;
  auto zero_32bit = std::uint32_t{0};
  auto one_32bit = std::uint32_t{1};
  auto zero_64bit = std::size_t{0};
  auto one_64bit = std::size_t{1};

  ASSERT_EQ(detail::index_type<true>{}, zero);
  ASSERT_NE(detail::index_type<true>{}, one);
  ASSERT_EQ(detail::index_type<true>{}, zero_32bit);
  ASSERT_NE(detail::index_type<true>{}, one_32bit);
  ASSERT_EQ(detail::index_type<true>{}, zero_64bit);
  ASSERT_NE(detail::index_type<true>{}, one_64bit);

  ASSERT_EQ(detail::index_type<true>{one_32bit}, one);
  ASSERT_NE(detail::index_type<true>{one_32bit}, zero);
  ASSERT_EQ(detail::index_type<true>{one_32bit}, one_32bit);
  ASSERT_NE(detail::index_type<true>{one_32bit}, zero_32bit);
  ASSERT_EQ(detail::index_type<true>{one_32bit}, one_64bit);
  ASSERT_NE(detail::index_type<true>{one_32bit}, zero_64bit);

  ASSERT_EQ(detail::index_type<true>{one_64bit}, one);
  ASSERT_NE(detail::index_type<true>{one_64bit}, zero);
  ASSERT_EQ(detail::index_type<true>{one_64bit}, one_32bit);
  ASSERT_NE(detail::index_type<true>{one_64bit}, zero_32bit);
  ASSERT_EQ(detail::index_type<true>{one_64bit}, one_64bit);
  ASSERT_NE(detail::index_type<true>{one_64bit}, zero_64bit);
}

TEST(FilBackend, diff_type) {
  auto zero = 0;
  auto neg_one = -1;
  auto zero_32bit = std::int32_t{0};
  auto neg_one_32bit = std::int32_t{-1};
  auto zero_64bit = std::ptrdiff_t{0};
  auto neg_one_64bit = std::ptrdiff_t{-1};

  ASSERT_EQ(detail::diff_type<true>{}, zero);
  ASSERT_NE(detail::diff_type<true>{}, neg_one);
  ASSERT_EQ(detail::diff_type<true>{}, zero_32bit);
  ASSERT_NE(detail::diff_type<true>{}, neg_one_32bit);
  ASSERT_EQ(detail::diff_type<true>{}, zero_64bit);
  ASSERT_NE(detail::diff_type<true>{}, neg_one_64bit);

  ASSERT_EQ(detail::diff_type<true>{neg_one_32bit}, neg_one);
  ASSERT_NE(detail::diff_type<true>{neg_one_32bit}, zero);
  ASSERT_EQ(detail::diff_type<true>{neg_one_32bit}, neg_one_32bit);
  ASSERT_NE(detail::diff_type<true>{neg_one_32bit}, zero_32bit);
  ASSERT_EQ(detail::diff_type<true>{neg_one_32bit}, neg_one_64bit);
  ASSERT_NE(detail::diff_type<true>{neg_one_32bit}, zero_64bit);

  ASSERT_EQ(detail::diff_type<true>{neg_one_64bit}, neg_one);
  ASSERT_NE(detail::diff_type<true>{neg_one_64bit}, zero);
  ASSERT_EQ(detail::diff_type<true>{neg_one_64bit}, neg_one_32bit);
  ASSERT_NE(detail::diff_type<true>{neg_one_64bit}, zero_32bit);
  ASSERT_EQ(detail::diff_type<true>{neg_one_64bit}, neg_one_64bit);
  ASSERT_NE(detail::diff_type<true>{neg_one_64bit}, zero_64bit);
}

}

