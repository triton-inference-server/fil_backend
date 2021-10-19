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

#include <names.h>
#include <serialization.h>
#include <tl_utils.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <rapids_triton/model/shared_state.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {

auto constexpr DEFAULT_TRANSFER_THRESHOLD = std::size_t{};

struct RapidsSharedState : rapids::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : rapids::SharedModelState{std::move(config), true}
  {
  }

  void load()
  {
    predict_proba_ = get_config_param<bool>("predict_proba", false);
    model_format_ = string_to_serialization(
        get_config_param<std::string>("model_type", std::string{"xgboost"}));
    transfer_threshold_ = get_config_param<std::size_t>(
        "transfer_threshold", DEFAULT_TRANSFER_THRESHOLD);

    tl_params_->algo = name_to_tl_algo(
        get_config_param<std::string>("algo", std::string("ALGO_AUTO")));
    tl_params_->storage_type = name_to_storage_type(
        get_config_param<std::string>("storage_type", std::string("AUTO")));
    tl_params_->output_class = get_config_param<bool>("output_class");
    if (tl_params_->output_class) {
      tl_params_->threshold = get_config_param<float>("threshold");
    } else {
      tl_params_->threshold = 0.5f;
    }
    tl_params_->blocks_per_sm = get_config_param<int>("blocks_per_sm", 0);
    tl_params_->threads_per_tree =
        std::max(1, get_config_param<int>("threads_per_tree", 1));
  }

  auto predict_proba() const { return predict_proba_; }
  auto model_format() const { return model_format_; }
  auto transfer_threshold() const { return transfer_threshold_; }
  auto treelite_params() const { return tl_params_; }

 private:
  bool predict_proba_{};
  SerializationFormat model_format_{};
  std::size_t transfer_threshold_{};
  std::shared_ptr<ML::fil::treelite_params_t> tl_params_ =
      std::make_shared<ML::fil::treelite_params_t>();
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
