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

#include <detail/postprocess.h>
#include <names.h>

#include <cstddef>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <>
struct ClassEncoder<rapids::HostMemory> {
  ClassEncoder() : nthread_(1) {}
  ClassEncoder(int nthread) : nthread_(nthread) {}

  void argmax_for_multiclass(
      rapids::Buffer<float>& output, rapids::Buffer<float>& input,
      std::size_t samples, std::size_t num_classes) const
  {
    // Perform argmax for multi-class classification
    auto* dest = output.data();
    auto* src = input.data();
#pragma omp parallel for num_threads(nthread_)
    for (std::size_t i = 0; i < samples; ++i) {
      float max_prob = 0.0f;
      int max_class = 0;
      for (std::size_t j = 0; j < num_classes; ++j) {
        if (src[i * num_classes + j] > max_prob) {
          max_prob = src[i * num_classes + j];
          max_class = j;
        }
      }
      dest[i] = max_class;
    }
  }
  void threshold_inplace(
      rapids::Buffer<float>& output, std::size_t samples, float threshold) const
  {
    // Perform thresholding in-place for binary classification
    auto* out = output.data();
#pragma omp parallel for num_threads(nthread_)
    for (std::size_t i = 0; i < samples; ++i) {
      out[i] = (out[i] > threshold) ? 1.0f : 0.0f;
    }
  }

 private:
  int nthread_;
};

}}}  // namespace triton::backend::NAMESPACE
