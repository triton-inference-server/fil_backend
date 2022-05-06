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

#include <gtest/gtest.h>
#include <kayak/detail/host_only_throw.hpp>
#include <kayak/gpu_support.hpp>
#include <stdexcept>

namespace kayak {
TEST(FilBackend, host_only_throw_gpu) {
  if constexpr (!GPU_COMPILATION) {
    host_only_throw<std::runtime_error>("Expected exception");
  }
}
}
