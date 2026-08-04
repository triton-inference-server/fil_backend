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

#include <detail/postprocess.h>
#include <tl_model.h>

#ifdef TRITON_ENABLE_GPU
#include <detail/postprocess_gpu.h>
#endif
#include <detail/postprocess_cpu.h>

#include <cstddef>
#include <nvforest/forest_model.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton::backend { namespace NAMESPACE { namespace detail {

template <rapids::MemoryType memory_type>
void
nvforest_predict(
    nvforest::forest_model& nvforest_model,
    nvforest::handle_t const& nvforest_handle,
    TreeliteModel const& treelite_model, rapids::Buffer<float>& output,
    rapids::Buffer<float const> const& input, std::size_t samples,
    bool predict_proba)
{
  // Assume: input and output reside on `memory_type` (host or device)
  if (output.mem_type() != memory_type || input.mem_type() != memory_type) {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        std::string{"input and output arrays must reside on "} +
            rapids::memory_type_to_string(memory_type));
  }

  // Create non-owning Buffer to same memory as `output`
  auto output_buffer = rapids::Buffer<float>{
      output.data(), output.size(), output.mem_type(), output.device(),
      output.stream()};
  auto output_size = output.size();
  // nvForest expects buffer of size samples * num_classes for multi-class
  // classifiers, but output buffer may be smaller, so we need a temporary
  // buffer
  auto const num_classes = treelite_model.num_classes();
  if (!predict_proba && treelite_model.config().is_classifier &&
      num_classes > 1) {
    nvforest_model.set_row_postprocessing(nvforest::row_op::max_index);
    output_size = samples * num_classes;
    if (output_size != output.size()) {
      // If expected output size is not the same as the size of `output`,
      // create a temporary buffer of the correct size
      output_buffer = rapids::Buffer<float>{
          output_size, memory_type, output.device(), output.stream()};
    }
  }
  // For some binary classifiers, nvForest will output a single probability
  // score per input, but the client may be expecting two probability
  // scores (for positive and negative classes). In this case,
  // a temp buffer is necessary.
  bool convert_binary_probs = false;
  if (predict_proba && treelite_model.config().is_classifier &&
      num_classes == 1 && output.size() == samples * 2) {
    output_buffer = rapids::Buffer<float>{
        samples, memory_type, output.device(), output.stream()};
    convert_binary_probs = true;
  }

  auto mem_type = (memory_type == rapids::DeviceMemory)
                      ? nvforest::device_type::gpu
                      : nvforest::device_type::cpu;
  nvforest_model.predict(
      nvforest_handle, output_buffer.data(), const_cast<float*>(input.data()),
      samples, mem_type, mem_type, nvforest::infer_kind::default_kind,
      treelite_model.config().chunk_size);
  nvforest_handle.synchronize();
  output_buffer.stream_synchronize();

  if (!predict_proba && treelite_model.config().is_classifier) {
    if (num_classes > 1) {
      // Multi-class classifiers
      // Gather class outputs into the output array by setting
      //   output[i] := output_buffer[i * num_classes]
      multiclass_classifier::gather_class_output<memory_type>(
          samples, num_classes, output, output_buffer);
    } else if (num_classes == 1) {
      // Binary classifiers (predict_proba=False):
      // Apply thresholding to convert probability scores to class predictions
      //   output[i] := (1 if output[i] > threshold else 0)
      binary_classifier::convert_probability_to_class<memory_type>(
          samples, output, treelite_model.config().threshold);
    }
  }
  // Binary classifiers (predict_proba=True):
  // Convert (n, 1) probability score matrix to (n, 2) matrix
  //   output[i, 0] := 1 - output_buffer[i, 0]
  //   output[i, 1] := output_buffer[i, 0]
  if (convert_binary_probs) {
    binary_classifier::convert_probability<memory_type>(
        samples, output, output_buffer);
  }
}

}}}  // namespace triton::backend::NAMESPACE::detail
