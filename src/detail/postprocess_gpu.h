/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

namespace triton { namespace backend { namespace NAMESPACE { namespace detail {

namespace binary_classifier {

// Apply thresholding to convert probability scores to class predictions
//   output[i] := (1 if output[i] > threshold else 0)
void convert_probability_to_class(
    std::size_t n_samples, rapids::Buffer<float>& output, float threshold);

// Convert (n, 1) probability score matrix to (n, 2) matrix
//   output[i, 0] := 1 - input[i, 0]
//   output[i, 1] := input[i, 0]
void convert_probability(
    std::size_t n_samples, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input);

}  // namespace binary_classifier

namespace multiclass_classifier {

// Gather class outputs into the output array by setting
//   output[i] := input[i * n_classes]
void gather_class_output(
    std::size_t n_samples, std::size_t n_classes, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input);

}  // namespace multiclass_classifier

void print_buffer(rapids::Buffer<float> const& buffer);
void print_buffer(rapids::Buffer<float const> const& buffer);

}}}}  // namespace triton::backend::NAMESPACE::detail
