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
#include <treelite/logging.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <rapids_triton/exceptions.hpp>
#include <sstream>

namespace triton { namespace backend { namespace NAMESPACE {

inline auto
load_tl_base_model(
    std::filesystem::path const& model_file, SerializationFormat format,
    bool xgboost_allow_unknown_field)
{
  auto result = std::unique_ptr<treelite::Model>{};

  try {
    switch (format) {
      case SerializationFormat::xgboost:
        result = treelite::model_loader::LoadXGBoostModelLegacyBinary(
            model_file.c_str());
        break;
      case SerializationFormat::xgboost_json: {
        auto config_str =
            std::string("{\"allow_unknown_field\": ") +
            std::string(xgboost_allow_unknown_field ? "true" : "false") + "}";
        result = treelite::model_loader::LoadXGBoostModel(
            model_file.c_str(), config_str.c_str());
        break;
      }
      case SerializationFormat::lightgbm:
        result = treelite::model_loader::LoadLightGBMModel(model_file.c_str());
        break;
      case SerializationFormat::treelite: {
        auto file = std::fstream{model_file.c_str()};
        try {
          if (file.is_open()) {
            result = treelite::Model::DeserializeFromStream(file);
          } else {
            auto log_stream = std::stringstream{};
            log_stream << "Could not open model file " << model_file;
            throw rapids::TritonException(
                rapids::Error::Unavailable, log_stream.str());
          }
        }
        catch (treelite::Error const& err) {
          throw;
        }
        break;
      }
    }
  }
  catch (treelite::Error const& err) {
    throw rapids::TritonException(rapids::Error::Unknown, err.what());
  }
  catch (std::runtime_error const& err) {
    // This block is needed because Treelite sometimes throws a generic
    // exception
    // TODO(hcho3): Revise Treelite so that it only throws treelite::Error
    throw rapids::TritonException(rapids::Error::Unknown, err.what());
  }

  return result;
}

inline auto
tl_get_num_classes(treelite::Model const& base_model)
{
  TREELITE_CHECK_EQ(base_model.num_target, 1)
      << "Multi-target model not supported";
  return base_model.num_class[0];
}

}}}  // namespace triton::backend::NAMESPACE
