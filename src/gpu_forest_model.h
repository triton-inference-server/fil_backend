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
#include <cuml/fil/fil.h>
#include <fil_config.h>
#include <forest_model.h>
#include <names.h>
#include <tl_model.h>

#include <cstddef>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/handle.hpp>
#include <cuml/experimental/fil/forest_model.hpp>
#include <cuml/experimental/fil/treelite_importer.hpp>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

using fil_forest_t = ML::fil::forest_t<float>;
namespace filex = ML::experimental::fil;

template <>
struct ForestModel<rapids::DeviceMemory> {
  using device_id_t = int;
  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model, bool use_new_fil)
      : device_id_{device_id}, raft_handle_{stream}, tl_model_{tl_model},
        fil_forest_{[this]() {
          auto result = fil_forest_t{};
          auto variant_result = ML::fil::forest_variant{};
          auto config = tl_to_fil_config(tl_model_->config());
          ML::fil::from_treelite(
              raft_handle_, &variant_result, tl_model_->handle(), &config);
          try {
            result = std::get<fil_forest_t>(variant_result);
          }
          catch (std::bad_variant_access const& err) {
            throw rapids::TritonException(
                rapids::Error::Internal,
                "Model did not load with expected precision");
          }
          return result;
        }()},
        new_fil_model_{[this, use_new_fil]() {
          auto result = std::optional<filex::forest_model>{};
          if (use_new_fil) {
            try {
              result = filex::import_from_treelite_model(
                  *tl_model_->base_tl_model(), filex::preferred_tree_layout,
                  filex::index_type{}, std::nullopt,
                  raft_proto::device_type::gpu);
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
        }()}
  {
  }

  ForestModel(ForestModel const& other) = default;
  ForestModel& operator=(ForestModel const& other) = default;
  ForestModel(ForestModel&& other) = default;
  ForestModel& operator=(ForestModel&& other) = default;

  ~ForestModel() noexcept { ML::fil::free(raft_handle_, fil_forest_); }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    if (new_fil_model_) {
      // TODO(hcho3): Revise new FIL so that it takes in (const io_t*) type for
      // input buffer
      new_fil_model_->predict(
          raft_proto::handle_t{raft_handle_}, output.data(),
          const_cast<float*>(input.data()), samples,
          get_raft_proto_device_type(output.mem_type()),
          get_raft_proto_device_type(input.mem_type()),
          filex::infer_kind::default_kind);
      // TODO(hcho3): handle predict_proba
    } else {
      ML::fil::predict(
          raft_handle_, fil_forest_, output.data(), input.data(), samples,
          predict_proba);
    }
  }

 private:
  raft::handle_t raft_handle_;
  std::shared_ptr<TreeliteModel> tl_model_;
  fil_forest_t fil_forest_;
  device_id_t device_id_;
  // TODO(hcho3): Make filex::forest_model::predict() a const method
  mutable std::optional<filex::forest_model> new_fil_model_;
};

}}}  // namespace triton::backend::NAMESPACE
