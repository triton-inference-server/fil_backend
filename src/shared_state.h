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
#include <tl_config.h>
#include <tl_utils.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <rapids_triton/model/shared_state.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

auto constexpr DEFAULT_TRANSFER_THRESHOLD = std::size_t{};

struct RapidsSharedState : rapids::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : rapids::SharedModelState{std::move(config), true}
  {
  }

  void load()
  {
    std::optional<bool> deprecated_output_class_param{std::nullopt};
    /* Handle parameters from old FIL */
    for (auto const& [removed_param, new_param] :
         std::vector<std::pair<std::string, std::string>>{
             {"algo", "layout"},
             {"threads_per_tree", "chunk_size"},
             {"output_class", "is_classifier"}}) {
      auto removed_param_value =
          get_config_param<std::string>(removed_param, std::string{});
      if (!removed_param_value.empty()) {
        rapids::log_warn(__FILE__, __LINE__)
            << "The `" << removed_param
            << "` parameter has been removed in 25.09 release. "
            << "Use `" << new_param << "` instead.";
        if (removed_param == "output_class") {
          if (removed_param_value == "true") {
            deprecated_output_class_param = true;
          } else if (removed_param_value == "false") {
            deprecated_output_class_param = false;
          }
        }
      }
    }
    for (auto const& removed_param :
         std::vector<std::string>{"storage_type", "blocks_per_sm"}) {
      if (!get_config_param<std::string>(removed_param, std::string{})
               .empty()) {
        rapids::log_warn(__FILE__, __LINE__)
            << "The `" << removed_param
            << "` parameter has been removed in 25.09 release.";
      }
    }

    predict_proba_ = get_config_param<bool>("predict_proba", false);
    model_format_ = string_to_serialization(
        get_config_param<std::string>("model_type", std::string{"xgboost"}));
    xgboost_allow_unknown_field_ =
        get_config_param<bool>("xgboost_allow_unknown_field", false);
    transfer_threshold_ = get_config_param<std::size_t>(
        "transfer_threshold", DEFAULT_TRANSFER_THRESHOLD);

    tl_config_->layout =
        get_config_param<std::string>("layout", std::string("depth_first"));
    if (deprecated_output_class_param.has_value()) {
      tl_config_->is_classifier = deprecated_output_class_param.value();
    } else {
      tl_config_->is_classifier = get_config_param<bool>("is_classifier");
    }
    if (tl_config_->is_classifier) {
      tl_config_->threshold = get_config_param<float>("threshold");
    } else {
      tl_config_->threshold = 0.5f;
    }
    tl_config_->chunk_size =
        std::max(1, get_config_param<int>("chunk_size", 1));
    tl_config_->cpu_nthread = get_config_param<int>("cpu_nthread", -1);
    use_herring_ =
        get_config_param<bool>("use_experimental_optimizations", false);
  }

  auto predict_proba() const { return predict_proba_; }
  auto model_format() const { return model_format_; }
  auto xgboost_allow_unknown_field() const
  {
    return xgboost_allow_unknown_field_;
  }
  auto transfer_threshold() const { return transfer_threshold_; }
  auto config() const { return tl_config_; }
  auto use_herring() const { return use_herring_; }

 private:
  bool predict_proba_{};
  SerializationFormat model_format_{};
  bool xgboost_allow_unknown_field_{};
  std::size_t transfer_threshold_{};
  std::shared_ptr<treelite_config> tl_config_ =
      std::make_shared<treelite_config>();
  bool use_herring_{};
};

}}}  // namespace triton::backend::NAMESPACE
