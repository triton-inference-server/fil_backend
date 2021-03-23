#include <treelite/c_api.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>

#include <triton_fil/config.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    common::TritonJson::Value&& model_config)
    : BackendModel(triton_model), treelite_handle(nullptr),
      triton_model_(triton_model), name_(name), version_(version),
      model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false)
{
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  triton::common::TritonJson::Value config;

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));

  (*state)->ModelConfig().Find("parameters", &config);
  RETURN_IF_ERROR(tl_params_from_config(config, (*state)->tl_params));
  return nullptr;  // success
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
