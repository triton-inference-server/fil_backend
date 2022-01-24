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
#include <cuml/fil/fil.h>
#include <names.h>
#include <tl_config.h>

#include <rapids_triton/exceptions.hpp>
#include <sstream>

namespace triton {
namespace backend {
namespace NAMESPACE {

namespace detail {

inline auto name_to_tl_algo(std::string const& name)
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

inline auto name_to_storage_type(std::string const& name)
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

}  // namespace detail

inline auto tl_to_fil_config(treelite_config const& tl_config)
{
  return ML::fil::treelite_params_t{detail::name_to_tl_algo(tl_config.algo),
                                    tl_config.output_class,
                                    tl_config.threshold,
                                    detail::name_to_storage_type(tl_config.storage_type),
                                    tl_config.blocks_per_sm,
                                    tl_config.threads_per_tree,
                                    0,
                                    nullptr};
}

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
