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
#include <treelite/c_api.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <rapids_triton/exceptions.hpp>
#include <sstream>

namespace triton { namespace backend { namespace NAMESPACE {

inline auto*
load_tl_handle(
    std::filesystem::path const& model_file, SerializationFormat format)
{
  auto handle = static_cast<void*>(nullptr);

  auto load_result = int{};

  switch (format) {
    case SerializationFormat::xgboost:
      load_result = TreeliteLoadXGBoostModel(model_file.c_str(), &handle);
      break;
    case SerializationFormat::xgboost_json:
      load_result = TreeliteLoadXGBoostJSON(model_file.c_str(), &handle);
      break;
    case SerializationFormat::lightgbm:
      load_result = TreeliteLoadLightGBMModel(model_file.c_str(), &handle);
      break;
    case SerializationFormat::treelite:
      load_result = TreeliteDeserializeModel(model_file.c_str(), &handle);
      break;
  }
  if (load_result != 0) {
    auto log_stream = std::stringstream{};
    log_stream << "Model failed to load into Treelite with error: "
               << TreeliteGetLastError();
    throw rapids::TritonException(rapids::Error::Unknown, log_stream.str());
  }

  return handle;
}

inline auto
tl_get_num_classes(void* handle)
{
  auto result = std::size_t{};
  if (TreeliteQueryNumClass(handle, &result) != 0) {
    throw rapids::TritonException(
        rapids::Error::Unknown,
        "Treelite could not determine number of classes in model");
  }

  return result;
}

inline auto
name_to_tl_algo(std::string const& name)
{
  auto result = ML::fil::algo_t{};
  if (name == "ALGO_AUTO") {
    result = ML::fil::algo_t::ALGO_AUTO;
  } else if (name == "NAIVE") {
    result = ML::fil::algo_t::NAIVE;
  } else if (name == "TREE_REORG") {
    result = ML::fil::algo_t::TREE_REORG;
  } else if (name == "BATCH_TREE_REORG") {
    result = ML::fil::algo_t::BATCH_TREE_REORG;
  } else {
    auto log_stream = std::stringstream{};
    log_stream << "Unknown FIL algorithm name: " << name;
    throw rapids::TritonException(rapids::Error::InvalidArg, log_stream.str());
  }

  return result;
}

inline auto
name_to_storage_type(std::string const& name)
{
  auto result = ML::fil::storage_type_t{};
  if (name == "AUTO") {
    result = ML::fil::storage_type_t::AUTO;
  } else if (name == "DENSE") {
    result = ML::fil::storage_type_t::DENSE;
  } else if (name == "SPARSE") {
    result = ML::fil::storage_type_t::SPARSE;
  } else if (name == "SPARSE8") {
    result = ML::fil::storage_type_t::SPARSE8;
  } else {
    auto log_stream = std::stringstream{};
    log_stream << "Unknown FIL storage type name: " << name;
    throw rapids::TritonException(rapids::Error::InvalidArg, log_stream.str());
  }

  return result;
}

}}}  // namespace triton::backend::NAMESPACE
