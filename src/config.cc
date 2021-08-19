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

#include <cuml/fil/fil.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/config.h>
#include <triton_fil/enum_conversions.h>

#include <exception>
#include <optional>
#include <string>

namespace triton { namespace backend { namespace fil {

ML::fil::treelite_params_t
tl_params_from_config(triton::common::TritonJson::Value& config)
{
  ML::fil::treelite_params_t out_params;
  out_params.algo = name_to_tl_algo(retrieve_param<std::string>(
      config, "algo", std::optional<std::string>(std::string("ALGO_AUTO"))));
  out_params.storage_type = name_to_storage_type(retrieve_param<std::string>(
      config, "storage_type", std::optional<std::string>(std::string("AUTO"))));
  out_params.output_class = retrieve_param<bool>(config, "output_class");
  if (out_params.output_class) {
    out_params.threshold = retrieve_param<float>(config, "threshold");
  } else {
    out_params.threshold = 0.5;
  }
  out_params.blocks_per_sm =
      retrieve_param<int>(config, "blocks_per_sm", std::optional<int>(0));
  out_params.threads_per_tree =
      retrieve_param<int>(config, "threads_per_tree", std::optional<int>(1));

  return out_params;
};

}}}  // namespace triton::backend::fil
