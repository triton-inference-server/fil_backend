/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <herring2/gpu_support.hpp>

namespace herring {

TEST(FilBackend, gpu_support)
{
#ifdef ENABLE_GPU
  ASSERT_EQ(GPU_ENABLED, true) << "GPU_ENABLED constant has wrong value\n";
#else
  ASSERT_EQ(GPU_ENABLED, false) << "GPU_ENABLED constant has wrong value\n";
#endif
}
}  // namespace triton

