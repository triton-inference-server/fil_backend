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

#include <cuda_runtime_api.h>
#include <detail/postprocess_gpu.h>
#include <forest_model.h>
#include <names.h>
#include <nvforest_config.h>
#include <nvforest_predict.h>
#include <tl_model.h>

#include <cstddef>
#include <memory>
#include <nvforest/cuda_stream.hpp>
#include <nvforest/device_type.hpp>
#include <nvforest/forest_model.hpp>
#include <nvforest/handle.hpp>
#include <nvforest/treelite_importer.hpp>
#include <raft/core/handle.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton::backend { namespace NAMESPACE {

template <>
struct ForestModel<rapids::DeviceMemory> {
  using device_id_t = int;
  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model)
      : device_id_{device_id}, raft_handle_{stream}, tl_model_{tl_model},
        nvforest_model_{[this, device_id, stream]() {
          auto config = tl_model_->config();
          auto result = nvforest::import_from_treelite_handle(
              tl_model_->handle(),
              detail::name_to_nvforest_layout(config.layout), 128, false,
              nvforest::device_type::gpu, device_id,
              nvforest::cuda_stream{stream});
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
    detail::nvforest_predict<rapids::DeviceMemory>(
        nvforest_model_, nvforest::handle_t{raft_handle_}, *tl_model_, output,
        input, samples, predict_proba);
  }

 private:
  raft::handle_t raft_handle_;
  std::shared_ptr<TreeliteModel> tl_model_;
  mutable nvforest::forest_model nvforest_model_;
  device_id_t device_id_;
};

}}  // namespace triton::backend::NAMESPACE
