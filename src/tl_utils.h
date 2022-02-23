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
#include <treelite/frontend.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <rapids_triton/exceptions.hpp>
#include <sstream>

namespace triton { namespace backend { namespace NAMESPACE {

inline auto load_tl_base_model(
    std::filesystem::path const& model_file, SerializationFormat format)
{
  auto result = std::unique_ptr<treelite::Model>{};

  try {
    switch (format) {
      case SerializationFormat::xgboost:
        result = treelite::frontend::LoadXGBoostModel(model_file.c_str());
        break;
      case SerializationFormat::xgboost_json:
        result = treelite::frontend::LoadXGBoostJSONModel(model_file.c_str());
        break;
      case SerializationFormat::lightgbm:
        result = treelite::frontend::LoadLightGBMModel(model_file.c_str());
        break;
      case SerializationFormat::treelite: {
        auto* file = std::fopen(model_file.c_str(), "rb");
        if (file == nullptr) {
          auto log_stream = std::stringstream{};
          log_stream << "Could not open model file " << model_file;
          throw rapids::TritonException(rapids::Error::Unavailable, log_stream.str());
        }
        try {
          result = treelite::Model::DeserializeFromFile(file);
        } catch (treelite::Error const& err) {
          std::fclose(file);
          throw;
        }
        std::fclose(file);
        break;
      }
    }
  } catch (treelite::Error const& err) {
    throw rapids::TritonException(rapids::Error::Unknown, err.what());
  }

  return result;
}

inline auto
tl_get_num_classes(treelite::Model const& base_model)
{
  return base_model.task_param.num_class;
}

}}}  // namespace triton::backend::NAMESPACE
