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

#pragma once

#include <cuda_runtime_api.h>
#include <detail/postprocess_gpu.h>
#include <fil_config.h>
#include <forest_model.h>
#include <names.h>
#include <tl_model.h>

#include <cstddef>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/forest_model.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/treelite_importer.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <>
struct ForestModel<rapids::DeviceMemory> {
  using device_id_t = int;
  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model)
      : device_id_{device_id}, raft_handle_{stream}, tl_model_{tl_model},
        fil_forest_{[this, device_id, stream]() {
          auto config = tl_model_->config();
          auto result = ML::fil::import_from_treelite_handle(
              tl_model_->handle(), detail::name_to_fil_layout(config.layout),
              128, false, raft_proto::device_type::gpu, device_id, stream);
          return result;
        }()}
  {
  }

  ForestModel(ForestModel const& other) = default;
  ForestModel& operator=(ForestModel const& other) = default;
  ForestModel(ForestModel&& other) = default;
  ForestModel& operator=(ForestModel&& other) = default;

  ~ForestModel() noexcept {}

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    // Create non-owning Buffer to same memory as `output`
    auto output_buffer = rapids::Buffer<float>{
        output.data(), output.size(), output.mem_type(), output.device(),
        output.stream()};
    auto output_size = output.size();
    // FIL expects buffer of size samples * num_classes for multi-class
    // classifiers, but output buffer may be smaller, so we need a temporary
    // buffer
    auto const num_classes = tl_model_->num_classes();
    if (!predict_proba && tl_model_->config().is_classifier &&
        num_classes > 1) {
      output_size = samples * num_classes;
      if (output_size != output.size()) {
        // If expected output size is not the same as the size of `output`,
        // create a temporary buffer of the correct size
        output_buffer = rapids::Buffer<float>{
            output_size, rapids::DeviceMemory, output.device(),
            output.stream()};
      }
    }
    // For some binary classifiers, FIL will output a single probability
    // score per input, but the client may be expecting two probability
    // scores (for positive and negative classes). In this case,
    // a temp buffer is necessary.
    bool convert_binary_probs = false;
    if (predict_proba && tl_model_->config().is_classifier &&
        num_classes == 1 && output.size() == samples * 2) {
      output_buffer = rapids::Buffer<float>{
          samples, rapids::DeviceMemory, output.device(), output.stream()};
      convert_binary_probs = true;
    }

    fil_forest_.predict(
        raft_proto::handle_t{raft_handle_}, output_buffer.data(),
        const_cast<float*>(input.data()), samples, raft_proto::device_type::gpu,
        raft_proto::device_type::gpu, ML::fil::infer_kind::default_kind,
        tl_model_->config().chunk_size);
    raft_handle_.sync_stream();
    output_buffer.stream_synchronize();

    if (!predict_proba && tl_model_->config().is_classifier) {
      if (num_classes > 1) {
        // Multi-class classifiers
        // Gather class outputs into the output array by setting
        //   output[i] := output_buffer[i * num_classes]
        detail::multiclass_classifier::gather_class_output(
            samples, num_classes, output, output_buffer);
      } else if (num_classes == 1) {
        // Binary classifiers (predict_proba=False):
        // Apply thresholding to convert probability scores to class predictions
        //   output[i] := (1 if output[i] > threshold else 0)
        detail::binary_classifier::convert_probability_to_class(
            samples, output, tl_model_->config().threshold);
      }
    }
    // Binary classifiers (predict_proba=True):
    // Convert (n, 1) probability score matrix to (n, 2) matrix
    //   output[i, 0] := 1 - output_buffer[i, 0]
    //   output[i, 1] := output_buffer[i, 0]
    if (convert_binary_probs) {
      detail::binary_classifier::convert_probability(
          samples, output, output_buffer);
    }
  }

 private:
  raft::handle_t raft_handle_;
  std::shared_ptr<TreeliteModel> tl_model_;
  mutable ML::fil::forest_model fil_forest_;
  device_id_t device_id_;
};

}}}  // namespace triton::backend::NAMESPACE
