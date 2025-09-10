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
#include <cuml/fil/tree_layout.hpp>
#include <names.h>
#include <tl_config.h>

#include <rapids_triton/exceptions.hpp>
#include <sstream>

namespace triton { namespace backend { namespace NAMESPACE {

namespace detail {

inline auto
name_to_fil_layout(std::string const& name)
{
  auto result = ML::fil::tree_layout{};
  if (name == "depth_first") {
    result = ML::fil::tree_layout::depth_first;
  } else if (name == "breadth_first") {
    result = ML::fil::tree_layout::breadth_first;
  } else if (name == "layered" || name == "layered_children_together") {
    result = ML::fil::tree_layout::layered_children_together;
  } else {
    auto log_stream = std::stringstream{};
    log_stream << "Unknown FIL layout name: " << name;
    throw rapids::TritonException(rapids::Error::InvalidArg, log_stream.str());
  }

  return result;
}

}  // namespace detail

}}}  // namespace triton::backend::NAMESPACE
