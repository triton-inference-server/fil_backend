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

#include <forest_model.h>
#include <names.h>
#include <tl_model.h>

#include <cstddef>
#include <memory>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <>
struct ForestModel<rapids::HostMemory> {
  ForestModel() = default;
  ForestModel(std::shared_ptr<TreeliteModel> tl_model) : tl_model_{tl_model} {}

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    tl_model_->predict(output, input, samples, predict_proba);
  }


 private:
  std::shared_ptr<TreeliteModel> tl_model_;
};

}}}  // namespace triton::backend::NAMESPACE
