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
#include <memory>
#include <raft/handle.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <>
struct ForestModel<rapids::DeviceMemory> {
  using device_id_t = int;
  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model)
      : device_id_{device_id}, raft_handle_{}, tl_model_{tl_model},
        fil_forest_{[this]() {
          auto result = ML::fil::forest_t{};
          auto config = tl_to_fil_config(tl_model_->config());
          ML::fil::from_treelite(
              raft_handle_, &result, tl_model_->handle(), &config);
          return result;
        }()}
  {
    raft_handle_.set_stream(stream);
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
    ML::fil::predict(
        raft_handle_, fil_forest_, output.data(), input.data(), samples,
        predict_proba);
  }

 private:
  raft::handle_t raft_handle_;
  std::shared_ptr<TreeliteModel> tl_model_;
  ML::fil::forest_t fil_forest_;
  device_id_t device_id_;
};

}}}  // namespace triton::backend::NAMESPACE
