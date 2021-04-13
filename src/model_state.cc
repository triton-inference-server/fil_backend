/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <treelite/c_api.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton_fil/config.h>
#include <triton_fil/model_state.h>

#include <memory>

namespace triton { namespace backend { namespace fil {

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, const char* name, const uint64_t version)
    : BackendModel(triton_model), treelite_handle(nullptr)
{
}

std::unique_ptr<ModelState>
ModelState::Create(TRITONBACKEND_Model& triton_model)
{
  auto config = get_model_config(triton_model);
  std::string model_name = get_model_name(triton_model);
  uint64_t model_version = get_model_version(triton_model);
  auto state = std::make_unique<ModelState>(
      &triton_model, model_name.c_str(), model_version);

  state->ModelConfig().Find("parameters", config.get());
  // TODO: Properly handle tl_params in constructor
  state->tl_params = tl_params_from_config(*config);
  return state;
}

void
ModelState::LoadModel(
    std::string artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id)
{
  if (artifact_name.empty()) {
    artifact_name = "xgboost.model";
  }
  std::string model_path =
      JoinPath({RepositoryPath(), std::to_string(Version()), artifact_name});
  {
    bool is_dir;
    triton_check(IsDirectory(model_path, &is_dir));
    if (is_dir) {
      model_path = JoinPath({model_path, "xgboost.model"});
    }
  }

  {
    bool exists;
    triton_check(FileExists(model_path, &exists));
    if (!exists) {
      throw TritonException(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("unable to find '") + model_path +
              "' for model instance '" + Name() + "'");
    }
  }

  if (TreeliteLoadXGBoostModel(model_path.c_str(), &treelite_handle) != 0) {
    throw TritonException(
        TRITONSERVER_ERROR_UNAVAILABLE, "Treelite model could not be loaded");
  }
}

void
ModelState::UnloadModel()
{
  if (treelite_handle != nullptr) {
    TreeliteFreeModel(treelite_handle);
  }
}

}}}  // namespace triton::backend::fil
