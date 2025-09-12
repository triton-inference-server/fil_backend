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
#include <treelite/gtil.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <filesystem>
#include <herring/tl_helpers.hpp>
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
      bool use_herring, bool xgboost_allow_unknown_field)
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
        num_classes_{static_cast<size_t>(tl_get_num_classes(*base_tl_model_))},
        base_herring_model_{[this, use_herring]() {
          auto result = std::optional<herring::tl_dispatched_model>{};
          if (use_herring) {
            rapids::log_warn(__FILE__, __LINE__)
                << "use_experimental_optimizations option is deprecated. "
                << "It will be removed in the 25.10 release.";
            try {
              result = std::visit(
                  [&](auto&& concrete_model) {
                    return herring::convert_model(
                        *base_tl_model_, concrete_model);
                  },
                  base_tl_model_->variant_);
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
      // Create non-owning Buffer to same memory as `output`
      auto output_buffer = rapids::Buffer<float>{
          output.data(), output.size(), rapids::HostMemory};
      auto output_size = output.size();
      // GTIL expects buffer of size samples * num_classes_ for multi-class
      // classifiers, but output buffer may be smaller, so we need a temporary
      // buffer
      if (!predict_proba && tl_config_->is_classifier && num_classes_ > 1) {
        output_size = samples * num_classes_;
        if (output_size != output.size()) {
          // If expected output size is not the same as the size of `output`,
          // create a temporary buffer of the correct size
          output_buffer = rapids::Buffer<float>{
              output_size, rapids::HostMemory, output.device(),
              output.stream()};
        }
      }
      // For some binary classifiers, GTIL will output a single probability
      // score per input, but the client may be expecting two probability
      // scores (for positive and negative classes). In this case,
      // a temp buffer is necessary.
      bool convert_binary_probs = false;
      if (predict_proba && tl_config_->is_classifier && num_classes_ == 1 &&
          output.size() == samples * 2) {
        output_buffer = rapids::Buffer<float>{samples, rapids::HostMemory};
        convert_binary_probs = true;
      }

      auto gtil_config = treelite::gtil::Configuration{};
      gtil_config.nthread = tl_config_->cpu_nthread;

      try {
        treelite::gtil::Predict(
            *base_tl_model_, input.data(), samples, output_buffer.data(),
            gtil_config);
      }
      catch (treelite::Error const& tl_err) {
        throw rapids::TritonException(rapids::Error::Internal, tl_err.what());
      }

      if (!predict_proba && tl_config_->is_classifier) {
        if (num_classes_ > 1) {
          // Multi-class classifiers
          // Compute class output from probabilities
          //   output[i] := argmax(output_buffer[i * num_classes_ + j])
          float* dest = output.data();
          float* src = output_buffer.data();
          for (std::size_t i = 0; i < samples; ++i) {
            float max_prob = 0.0f;
            int max_class = 0;
            for (std::size_t j = 0; j < num_classes_; ++j) {
              if (src[i * num_classes_ + j] > max_prob) {
                max_prob = src[i * num_classes_ + j];
                max_class = j;
              }
            }
            dest[i] = max_class;
          }
        } else if (num_classes_ == 1) {
          // Binary classifiers (predict_proba=False):
          // Apply thresholding to convert probability scores to class
          // predictions
          //   output[i] := (1 if output[i] > threshold else 0)
          for (std::size_t i = 0; i < samples; ++i) {
            output.data()[i] =
                (output.data()[i] > tl_config_->threshold) ? 1.0f : 0.0f;
          }
        }
      }
      // Binary classifiers (predict_proba=True):
      // Convert (n, 1) probability score matrix to (n, 2) matrix
      //   output[i, 0] := 1 - output_buffer[i, 0]
      //   output[i, 1] := output_buffer[i, 0]
      if (convert_binary_probs) {
        float* dest = output.data();
        float* src = output_buffer.data();
        for (std::size_t i = 0; i < samples; ++i) {
          dest[i * 2] = 1.0 - src[i];
          dest[i * 2 + 1] = src[i];
        }
      }
    }
  }

 private:
  std::shared_ptr<treelite_config> tl_config_;
  std::unique_ptr<treelite::Model> base_tl_model_;
  std::size_t num_classes_;
  std::optional<herring::tl_dispatched_model> base_herring_model_;
};

}}}  // namespace triton::backend::NAMESPACE
