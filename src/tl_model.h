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
#include <names.h>
#include <serialization.h>
#include <tl_config.h>
#include <tl_utils.h>
#include <treelite/gtil.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>
#include <string>
#include <variant>

namespace triton { namespace backend { namespace NAMESPACE {
struct TreeliteModel {
  TreeliteModel(
      std::filesystem::path const& model_file, SerializationFormat format,
      std::shared_ptr<treelite_config> tl_config, bool predict_proba,
      bool xgboost_allow_unknown_field)
      : tl_config_{tl_config},
        base_tl_model_{[&model_file, &format, predict_proba,
                        xgboost_allow_unknown_field, this]() {
          auto result = load_tl_base_model(
              model_file, format, xgboost_allow_unknown_field);
          auto num_classes = tl_get_num_classes(*base_tl_model_);
          if (predict_proba &&
              result->task_type == treelite::TaskType::kMultiClf &&
              result->leaf_vector_shape[1] == 1) {
            result->postprocessor = "softmax";
          }
          if (predict_proba &&
              result->task_type == treelite::TaskType::kMultiClf &&
              result->leaf_vector_shape[1] > 1) {
            result->postprocessor = "identity_multiclass";
          }

          return result;
        }()},
        num_classes_{static_cast<size_t>(tl_get_num_classes(*base_tl_model_))}
  {
  }
  TreeliteModel(TreeliteModel const& other) = default;
  TreeliteModel& operator=(TreeliteModel const& other) = default;
  TreeliteModel(TreeliteModel&& other) = default;
  TreeliteModel& operator=(TreeliteModel&& other) = default;
  ~TreeliteModel() = default;

  auto base_tl_model() const { return base_tl_model_.get(); }
  auto* handle() const { return static_cast<void*>(base_tl_model_.get()); }
  auto num_classes() const { return num_classes_; }
  auto& config() const { return *tl_config_; }

 private:
  std::shared_ptr<treelite_config> tl_config_;
  std::unique_ptr<treelite::Model> base_tl_model_;
  std::size_t num_classes_;
};

}}}  // namespace triton::backend::NAMESPACE
