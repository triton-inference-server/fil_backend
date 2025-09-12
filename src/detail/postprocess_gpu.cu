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

#include <detail/postprocess_gpu.h>
#include <names.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>
#include <cstdint>

namespace triton { namespace backend { namespace NAMESPACE { namespace detail {

namespace binary_classifier {

void
convert_probability_to_class(
    std::size_t n_samples, rapids::Buffer<float>& output, float threshold)
{
  thrust::for_each(
      thrust::device, output.data(), output.data() + n_samples,
      [threshold] __device__(float& e) {
        return (e > threshold) ? 1.0f : 0.0f;
      });
}

void
convert_probability(
    std::size_t n_samples, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input)
{
  thrust::counting_iterator<std::size_t> cnt_iter =
      thrust::make_counting_iterator<std::size_t>(0);
  thrust::for_each(
      thrust::device, cnt_iter, cnt_iter + n_samples,
      [dest = output.data(), src = input.data()] __device__(std::size_t i) {
        dest[i * 2] = 1.0 - src[i];
        dest[i * 2 + 1] = src[i];
      });
}

}  // namespace binary_classifier

namespace multiclass_classifier {

void
gather_class_output(
    std::size_t n_samples, std::size_t n_classes, rapids::Buffer<float>& output,
    rapids::Buffer<float>& input)
{
  thrust::counting_iterator<std::size_t> cnt_iter =
      thrust::make_counting_iterator<std::size_t>(0);
  thrust::for_each(
      thrust::device, cnt_iter, cnt_iter + n_samples,
      [dest = output.data(), src = input.data(),
       n_classes] __device__(std::size_t i) { dest[i] = src[i * n_classes]; });
}

}  // namespace multiclass_classifier

void
print_buffer(rapids::Buffer<float> const& buffer)
{
  auto ptr = thrust::device_pointer_cast(buffer.data());
  thrust::device_vector<float> d(ptr, ptr + buffer.size());
  thrust::host_vector<float> h = d;
  for (float e : h) {
    std::cerr << e << ", ";
  }
  std::cerr << std::endl;
}

void
print_buffer(rapids::Buffer<float const> const& buffer)
{
  auto ptr = thrust::device_pointer_cast(buffer.data());
  thrust::device_vector<float> d(ptr, ptr + buffer.size());
  thrust::host_vector<float> h = d;
  for (float e : h) {
    std::cerr << e << ", ";
  }
  std::cerr << std::endl;
}

}}}}  // namespace triton::backend::NAMESPACE::detail
