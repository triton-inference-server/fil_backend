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

#include <detail/postprocess_cpu.h>
#include <forest_model.h>
#include <names.h>
#include <tl_model.h>

#include <algorithm>
#include <cstddef>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/handle.hpp>
#include <cuml/experimental/fil/forest_model.hpp>
#include <cuml/experimental/fil/treelite_importer.hpp>
#include <herring/omp_helpers.hpp>
#include <memory>
#include <optional>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

namespace filex = ML::experimental::fil;

template <>
struct ForestModel<rapids::HostMemory> {
  ForestModel() = default;
  using device_id_t = int;
  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model, bool use_new_fil)
      : device_id_{device_id}, tl_model_{tl_model},
        new_fil_model_{[this, use_new_fil]() {
          auto result = std::optional<filex::forest_model>{};
          if (use_new_fil) {
            try {
              result = filex::import_from_treelite_model(
                  *tl_model_->base_tl_model(), filex::preferred_tree_layout,
                  filex::index_type{}, std::nullopt,
                  raft_proto::device_type::cpu);
              rapids::log_info(__FILE__, __LINE__)
                  << "Loaded model to new FIL format";
            }
            catch (filex::model_import_error const& ex) {
              result = std::nullopt;
              auto log_stream = rapids::log_info(__FILE__, __LINE__);
              log_stream << "Experimental FIL load failed with error \"";
              log_stream << ex.what();
              log_stream << "\"; falling back to current FIL";
            }
          }
          return result;
        }()},
        class_encoder_{int(thread_count(tl_model_->config().cpu_nthread))}
  {
  }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    if (new_fil_model_) {
      // Create non-owning Buffer to same memory as `output`
      auto output_buffer = rapids::Buffer<float>{
          output.data(), output.size(), output.mem_type(), output.device(),
          output.stream()};
      auto output_size = output.size();
      // New FIL expects buffer of size samples * num_classes for multi-class
      // classifiers, but output buffer may be smaller, so we need a temporary
      // buffer
      auto const num_classes = tl_model_->num_classes();
      if (!predict_proba && tl_model_->config().output_class &&
          num_classes > 1) {
        output_size = samples * num_classes;
        if (output_size != output.size()) {
          // If expected output size is not the same as the size of `output`,
          // create a temporary buffer of the correct size
          output_buffer =
              rapids::Buffer<float>{output_size, rapids::HostMemory};
        }
      }

      // TODO(hcho3): Revise new FIL so that it takes in (const io_t*) type for
      // input buffer
      new_fil_model_->predict(
          raft_proto::handle_t{}, output_buffer.data(),
          const_cast<float*>(input.data()), samples,
          get_raft_proto_device_type(output.mem_type()),
          get_raft_proto_device_type(input.mem_type()),
          filex::infer_kind::default_kind);

      if (!predict_proba && tl_model_->config().output_class &&
          num_classes > 1) {
        class_encoder_.argmax_for_multiclass(
            output, output_buffer, samples, num_classes);
      } else if (
          !predict_proba && tl_model_->config().output_class &&
          num_classes == 1) {
        class_encoder_.threshold_inplace(
            output, samples, tl_model_->config().threshold);
      }
    } else {
      tl_model_->predict(output, input, samples, predict_proba);
    }
  }


 private:
  std::shared_ptr<TreeliteModel> tl_model_;
  device_id_t device_id_;
  // TODO(hcho3): Make filex::forest_model::predict() a const method
  mutable std::optional<filex::forest_model> new_fil_model_;
  ClassEncoder<rapids::HostMemory> class_encoder_;
};

}}}  // namespace triton::backend::NAMESPACE
