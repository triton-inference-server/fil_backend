/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton::backend { namespace NAMESPACE { namespace detail {

namespace binary_classifier {

// Apply thresholding to convert probability scores to class predictions
//   output[i] := (1 if output[i] > threshold else 0)
template <rapids::MemoryType M>
void
convert_probability_to_class(
    std::size_t n_samples, rapids::Buffer<float>& output, float threshold)
{
  throw rapids::TritonException(
      rapids::Error::Unsupported,
      "convert_probability_to_class invoked with a memory type unsupported "
      "by this build");
}

// Convert (n, 1) probability score matrix to (n, 2) matrix
//   output[i, 0] := 1 - input[i, 0]
//   output[i, 1] := input[i, 0]
template <rapids::MemoryType M>
void
convert_probability(
    std::size_t n_samples, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input)
{
  throw rapids::TritonException(
      rapids::Error::Unsupported,
      "convert_probability invoked with a memory type unsupported "
      "by this build");
}

}  // namespace binary_classifier

namespace multiclass_classifier {

// Gather class outputs into the output array by setting
//   output[i] := input[i * n_classes]
template <rapids::MemoryType M>
void
gather_class_output(
    std::size_t n_samples, std::size_t n_classes, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input)
{
  throw rapids::TritonException(
      rapids::Error::Unsupported,
      "gather_class_output invoked with a memory type unsupported "
      "by this build");
}

}  // namespace multiclass_classifier

}}}  // namespace triton::backend::NAMESPACE::detail
