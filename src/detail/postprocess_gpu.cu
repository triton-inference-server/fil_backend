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

#include <detail/postprocess_gpu.h>
#include <names.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>
#include <cstdint>

namespace triton { namespace backend { namespace NAMESPACE {

void
ClassEncoder<rapids::DeviceMemory>::argmax_for_multiclass(
    rapids::Buffer<float>& output, rapids::Buffer<float>& input,
    std::size_t samples, std::size_t num_classes) const
{
  // Perform argmax for multi-class classification
  thrust::counting_iterator<std::size_t> cnt_iter =
      thrust::make_counting_iterator<std::size_t>(0);
  thrust::for_each(
      thrust::device, cnt_iter, cnt_iter + samples,
      [dest = output.data(), src = input.data(),
       num_classes] __device__(std::size_t i) {
        float max_prob = 0.0f;
        int max_class = 0;
        for (std::size_t j = 0; j < num_classes; ++j) {
          if (src[i * num_classes + j] > max_prob) {
            max_prob = src[i * num_classes + j];
            max_class = j;
          }
        }
        dest[i] = max_class;
      });
}

void
ClassEncoder<rapids::DeviceMemory>::threshold_inplace(
    rapids::Buffer<float>& output, std::size_t samples, float threshold) const
{
  // Perform thresholding in-place for binary classification
  thrust::for_each(
      thrust::device, output.data(), output.data() + samples,
      [threshold] __device__(float& e) {
        return (e > threshold) ? 1.0f : 0.0f;
      });
}


}}}  // namespace triton::backend::NAMESPACE
