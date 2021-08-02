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
#include <triton/backend/backend_model.h>
#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/triton_utils.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace fil {
//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static std::unique_ptr<ModelState> Create(TRITONBACKEND_Model& triton_model);

  void LoadModel(
      std::string artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id);
  void UnloadModel();

  ML::fil::treelite_params_t tl_params;
  void* treelite_handle;
  std::string model_type;
  bool predict_proba;

  ModelState(
      TRITONBACKEND_Model* triton_model, const char* name,
      const uint64_t version);

  std::size_t num_class();
  std::vector<int64_t> get_output_shape(std::vector<int64_t> input_shape);

 private:
  std::size_t num_class_;
};

}}}  // namespace triton::backend::fil
