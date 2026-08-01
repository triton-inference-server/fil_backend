/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <detail/postprocess.h>
#include <names.h>

#include <cstddef>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton::backend { namespace NAMESPACE { namespace detail {

namespace binary_classifier {

template <>
void convert_probability_to_class<rapids::HostMemory>(
    std::size_t n_samples, rapids::Buffer<float>& output, float threshold);

template <>
void convert_probability<rapids::HostMemory>(
    std::size_t n_samples, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input);

}  // namespace binary_classifier

namespace multiclass_classifier {

template <>
void gather_class_output<rapids::HostMemory>(
    std::size_t n_samples, std::size_t n_classes, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input);

}  // namespace multiclass_classifier

}}}  // namespace triton::backend::NAMESPACE::detail
