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

#include <forest_model.h>
#include <names.h>
#include <nvforest_config.h>
#include <nvforest_predict.h>
#include <tl_model.h>

#include <cstddef>
#include <memory>
#include <nvforest/device_type.hpp>
#include <nvforest/forest_model.hpp>
#include <nvforest/handle.hpp>
#include <nvforest/treelite_importer.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton::backend { namespace NAMESPACE {

template <>
struct ForestModel<rapids::HostMemory> {
  ForestModel() = default;
  ForestModel(std::shared_ptr<TreeliteModel> tl_model)
      : tl_model_{tl_model}, nvforest_model_{[this]() {
          auto config = tl_model_->config();
          auto result = nvforest::import_from_treelite_handle(
              tl_model_->handle(),
              detail::name_to_nvforest_layout(config.layout), 128, false,
              nvforest::device_type::cpu);
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
    detail::nvforest_predict<rapids::HostMemory>(
        nvforest_model_, nvforest::handle_t{}, *tl_model_, output, input,
        samples, predict_proba);
  }


 private:
  std::shared_ptr<TreeliteModel> tl_model_;
  mutable nvforest::forest_model nvforest_model_;
};

}}  // namespace triton::backend::NAMESPACE
