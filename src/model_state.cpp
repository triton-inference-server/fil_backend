#include <treelite/c_api.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>

#include <memory>
#include <triton_fil/config.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, const char* name, const uint64_t version)
    : BackendModel(triton_model), treelite_handle(nullptr),
      triton_model_(triton_model), name_(name), version_(version)
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

TRITONSERVER_Error*
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
    RETURN_IF_ERROR(IsDirectory(model_path, &is_dir));
    if (is_dir) {
      model_path = JoinPath({model_path, "xgboost.model"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path +
            "' for model instance '" + Name() + "'");
  }

  if (TreeliteLoadXGBoostModel(model_path.c_str(), &treelite_handle) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
        "Treelite model could not be loaded");
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::UnloadModel()
{
  if (treelite_handle != nullptr) {
    TreeliteFreeModel(treelite_handle);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // TODO
  return nullptr;  // success
}

}}}  // namespace triton::backend::fil
