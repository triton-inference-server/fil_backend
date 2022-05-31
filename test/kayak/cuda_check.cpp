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

#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#endif
#include <gtest/gtest.h>

#include <kayak/cuda_check.hpp>
#include <kayak/exceptions.hpp>

namespace kayak {
TEST(FilBackend, cuda_check) {
#ifdef ENABLE_GPU
  EXPECT_THROW(cuda_check(cudaError::cudaErrorMissingConfiguration), bad_cuda_call);
  cuda_check(cudaSuccess);
#else
  cuda_check(5); // arbitrary input
#endif
}

}

