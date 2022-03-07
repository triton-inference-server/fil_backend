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
#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {
struct TreeliteModel {
  TreeliteModel(
      std::filesystem::path const& model_file, SerializationFormat format,
      std::shared_ptr<treelite_config> tl_config)
      : handle_{load_tl_handle(model_file, format)},
        num_classes_{tl_get_num_classes(handle_)}, tl_config_{tl_config}
  {
  }
  TreeliteModel(TreeliteModel const& other) = default;
  TreeliteModel& operator=(TreeliteModel const& other) = default;
  TreeliteModel(TreeliteModel&& other) = default;
  TreeliteModel& operator=(TreeliteModel&& other) = default;

  ~TreeliteModel() noexcept
  {
    if (handle_ != nullptr) {
      TreeliteFreeModel(handle_);
    }
  }

  auto* handle() const { return handle_; }
  auto num_classes() const { return num_classes_; }
  auto& config() const { return *tl_config_; }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    auto* handle_model = static_cast<treelite::Model*>(handle_);

    // Create non-owning Buffer to same memory as `output`
    auto output_buffer = rapids::Buffer<float>{
        output.data(), output.size(), output.mem_type(), output.device(),
        output.stream()};

    auto gtil_output_size = output.size();
    // GTIL expects buffer of size samples * num_classes_ for multi-class
    // classifiers, but output buffer may be smaller, so we will create a
    // temporary buffer
    if (!predict_proba && tl_config_->output_class && num_classes_ > 1) {
      gtil_output_size = samples * num_classes_;
      std::strcpy(handle_model->param.pred_transform, "max_index");
    }

    // If expected GTIL size is not the same as the size of `output`, create
    // a temporary buffer of the correct size
    if (gtil_output_size != output.size()) {
      output_buffer =
          rapids::Buffer<float>{gtil_output_size, rapids::HostMemory};
    }

    // Actually perform inference
    auto out_result_size = std::size_t{};
    auto gtil_result = TreeliteGTILPredict(
        handle_, input.data(), samples, output_buffer.data(),
        tl_config_->cpu_nthread, 1, &out_result_size);
    if (gtil_result != 0) {
      throw rapids::TritonException(
          rapids::Error::Internal, TreeliteGetLastError());
    }

    // Copy back to expected output location
    if (gtil_output_size != output.size()) {
      rapids::copy<float, float>(
          output, output_buffer, std::size_t{}, output.size());
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
  void* handle_;
  std::size_t num_classes_;
  std::shared_ptr<treelite_config> tl_config_;
};

}}}  // namespace triton::backend::NAMESPACE
