/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include <names.h>

#include <cstddef>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

/* This struct defines a unified interface for converting probabilities
 * to integer class outputs, on CPU and GPU targets. */
template <rapids::MemoryType M>
struct ClassEncoder {
  ClassEncoder() = default;
  void argmax_for_multiclass(
      rapids::Buffer<float>& output, rapids::Buffer<float>& input,
      std::size_t samples, std::size_t num_classes) const
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "ClassEncoder invoked with a memory type unsupported by this build");
  }
  void threshold_inplace(
      rapids::Buffer<float>& output, std::size_t samples, float threshold) const
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "ClassEncoder invoked with a memory type unsupported by this build");
  }
};

}}}  // namespace triton::backend::NAMESPACE
