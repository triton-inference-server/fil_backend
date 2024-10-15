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

#include <rapids_triton/exceptions.hpp>
#include <sstream>
#include <string>

namespace triton { namespace backend { namespace NAMESPACE {

enum struct SerializationFormat {
  xgboost,
  xgboost_json,
  xgboost_ubj,
  lightgbm,
  treelite
};

inline auto
string_to_serialization(std::string const& type_string)
{
  auto result = SerializationFormat{};

  if (type_string == "xgboost") {
    result = SerializationFormat::xgboost;
  } else if (type_string == "xgboost_json") {
    result = SerializationFormat::xgboost_json;
  } else if (type_string == "xgboost_ubj") {
    result = SerializationFormat::xgboost_ubj;
  } else if (type_string == "lightgbm") {
    result = SerializationFormat::lightgbm;
  } else if (type_string == "treelite_checkpoint") {
    result = SerializationFormat::treelite;
  } else {
    auto log_stream = std::stringstream{};
    log_stream << type_string
               << " not recognized as a valid serialization format.";
    throw rapids::TritonException(rapids::Error::Unsupported, log_stream.str());
  }

  return result;
}

inline auto
serialization_to_string(SerializationFormat format)
{
  auto result = std::string{};

  switch (format) {
    case SerializationFormat::xgboost:
      result = "xgboost";
      break;
    case SerializationFormat::xgboost_json:
      result = "xgboost_json";
      break;
    case SerializationFormat::xgboost_ubj:
      result = "xgboost_ubj";
      break;
    case SerializationFormat::lightgbm:
      result = "lightgbm";
      break;
    case SerializationFormat::treelite:
      result = "treelite_checkpoint";
      break;
  }

  return result;
}

}}}  // namespace triton::backend::NAMESPACE
