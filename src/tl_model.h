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
#include <treelite/frontend.h>
#include <treelite/gtil.h>
#include <treelite/tree.h>

#include <cstring>
#include <filesystem>
#include <herring/tl_helpers.hpp>
#include <memory>
#include <optional>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {
struct TreeliteModel {
  TreeliteModel(
      std::filesystem::path const& model_file, SerializationFormat format,
      std::shared_ptr<treelite_config> tl_config, bool predict_proba)
      : tl_config_{tl_config},
        base_tl_model_{[&model_file, &format, predict_proba, this]() {
          auto result = load_tl_base_model(model_file, format);
          auto num_classes = tl_get_num_classes(*base_tl_model_);
          if (!predict_proba && tl_config_->output_class && num_classes > 1) {
            std::strcpy(result->param.pred_transform, "max_index");
          }
          if (predict_proba &&
              result->task_type == treelite::TaskType::kMultiClfGrovePerClass) {
            std::strcpy(result->param.pred_transform, "softmax");
          }
          if (predict_proba &&
              result->task_type == treelite::TaskType::kMultiClfProbDistLeaf) {
            std::strcpy(result->param.pred_transform, "identity_multiclass");
          }

          return result;
        }()},
        num_classes_{tl_get_num_classes(*base_tl_model_)},
        base_herring_model_{[this]() {
          auto result = std::optional<herring::tl_dispatched_model>{};
          try {
            result = base_tl_model_->Dispatch([](auto const& concrete_model) {
              return herring::convert_model(concrete_model);
            });
            rapids::log_info(__FILE__, __LINE__)
                << "Loaded model to Herring format";
          }
          catch (herring::unconvertible_model_exception const& herring_err) {
            result = std::nullopt;
            auto log_stream = rapids::log_info(__FILE__, __LINE__);
            log_stream << "Herring load failed with error \"";
            log_stream << herring_err.what();
            log_stream << "\"; falling back to GTIL";
          }
          return result;
        }()}
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

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    // Create non-owning Buffer to same memory as `output`
    auto output_buffer = rapids::Buffer<float>{
        output.data(), output.size(), output.mem_type(), output.device(),
        output.stream()};

    if (base_herring_model_) {
      std::visit(
          [this, &input, &samples, &output_buffer](auto&& concrete_model) {
            concrete_model.predict(
                input.data(), samples, output_buffer.data(),
                tl_config_->cpu_nthread);
          },
          *base_herring_model_);
    } else {
      auto gtil_output_size = output.size();
      // GTIL expects buffer of size samples * num_classes_ for multi-class
      // classifiers, but output buffer may be smaller, so we will create a
      // temporary buffer
      if (!predict_proba && tl_config_->output_class && num_classes_ > 1) {
        gtil_output_size = samples * num_classes_;
      }

      // If expected GTIL size is not the same as the size of `output`, create
      // a temporary buffer of the correct size
      if (gtil_output_size != output.size()) {
        output_buffer =
            rapids::Buffer<float>{gtil_output_size, rapids::HostMemory};
      }

      auto gtil_config = treelite::gtil::Configuration{};
      gtil_config.nthread = tl_config_->cpu_nthread;
      auto gtil_output_shape =
          std::vector<std::size_t>{samples, output.size() / samples};

      // Actually perform inference
      try {
        treelite::gtil::Predict(
            base_tl_model_.get(), input.data(), samples, output_buffer.data(),
            gtil_config, gtil_output_shape);
      }
      catch (treelite::Error const& tl_err) {
        throw rapids::TritonException(rapids::Error::Internal, tl_err.what());
      }

      // Copy back to expected output location
      if (gtil_output_size != output.size()) {
        rapids::copy<float, float>(
            output, output_buffer, std::size_t{}, output.size());
      }
    }

    // Transform probabilities to desired output if necessary
    if (num_classes_ == 1 && predict_proba) {
      auto i = output.size();
      while (i > 0) {
        --i;
        output.data()[i] =
            ((i % 2) == 1) ? output.data()[i / 2] : 1.0f - output.data()[i / 2];
      }
    } else if (
        num_classes_ == 1 && !predict_proba && tl_config_->output_class) {
      std::transform(
          output.data(), output.data() + output.size(), output.data(),
          [this](float raw_pred) {
            return (raw_pred > tl_config_->threshold) ? 1.0f : 0.0f;
          });
    }
  }

 private:
  std::shared_ptr<treelite_config> tl_config_;
  std::unique_ptr<treelite::Model> base_tl_model_;
  std::size_t num_classes_;
  std::optional<herring::tl_dispatched_model> base_herring_model_;
};

}}}  // namespace triton::backend::NAMESPACE
