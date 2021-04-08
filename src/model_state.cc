// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
