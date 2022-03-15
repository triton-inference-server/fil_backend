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

#include <cuda_runtime_api.h>
#include <cuml/fil/fil.h>

#include <cstddef>
#include <memory>
#include <raft/handle.hpp>
#include <vector>

#include <matrix.hpp>

#include <run_fil.hpp>
#include <matrix.hpp>

ForestModel::ForestModel(std::unique_ptr<treelite::Model>& tl_model)
    : device_id_{}, raft_handle_{},
      fil_forest_{[this]() {
        auto result = ML::fil::forest_t{};
        auto config = ML::fil::treelite_params_t{
          ML::fil::algo_t::ALGO_AUTO,
          true,
          0.5,
          ML::fil::storage_type_t::AUTO,
          0,
          1,
          0,
          nullptr
        }
        ML::fil::from_treelite(
            raft_handle_, &result, static_cast<void*>(tl_model_.get()), &config);
        return result;
      }()}
{
}

ForestModel::ForestModel(ForestModel const& other) = default;
ForestModel::ForestModel& operator=(ForestModel const& other) = default;
ForestModel::ForestModel(ForestModel&& other) = default;
ForestModel::ForestModel& operator=(ForestModel&& other) = default;

ForestModel::~ForestModel() noexcept { ML::fil::free(raft_handle_, fil_forest_); }
void ForestModel::predict(float* output, matrix& input, bool predict_proba) const
{
  ML::fil::predict(
      raft_handle_, fil_forest_, output, input.data, input.rows,
      predict_proba);
}

void run_fil(ForestModel& model, matrix& input, std::vector<float>& output) {
  model.predict(output.data(), input, true);
}
