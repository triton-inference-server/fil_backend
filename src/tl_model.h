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
#include <xgboost/c_api.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>
#include <string>

namespace triton { namespace backend { namespace NAMESPACE {

void xgb_check(int err) {
  if (err != 0) {
    throw rapids::TritonException(rapids::Error::Internal, std::string{XGBGetLastError()});
  }
}

struct TreeliteModel {
  TreeliteModel(
      std::filesystem::path const& model_file, SerializationFormat format,
      std::shared_ptr<treelite_config> tl_config)
      : handle_{load_tl_handle(model_file, format)},
        num_classes_{tl_get_num_classes(handle_)}, tl_config_{tl_config}
  {
    xgb_check(XGBoosterCreate(nullptr, 0, &booster_));
    xgb_check(XGBoosterLoadModel(booster_, model_file.c_str()));
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
    xgb_check(XGBoosterFree(booster_));
  }

  auto* handle() const { return handle_; }
  auto num_classes() const { return num_classes_; }
  auto& config() const { return *tl_config_; }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    auto* handle_model = static_cast<treelite::Model*>(handle_);

    auto d_in = DMatrixHandle{};
    xgb_check(XGDMatrixCreateFromMat(input.data(), samples, input.size() / samples, NAN, &d_in));

    auto d_out_p = static_cast<float const*>(nullptr);
    auto out_size = static_cast<bst_ulong>(output.size());
    xgb_check(XGBoosterPredict(booster_, d_in, 0, 0, 0, &out_size, &d_out_p));

    xgb_check(XGDMatrixFree(d_in));

    std::copy(d_out_p, d_out_p + output.size(), output.data());
    // Transform probabilities to desired output if necessary
    /* if (num_classes_ == 1 && predict_proba) {
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
    } */
  }

 private:
  void* handle_;
  std::size_t num_classes_;
  std::shared_ptr<treelite_config> tl_config_;
  BoosterHandle booster_;
};

}}}  // namespace triton::backend::NAMESPACE
