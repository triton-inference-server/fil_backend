#pragma once

#include <memory>
#include <string>

#include <cuml/fil/fil.h>
#include <triton/backend/backend_model.h>
#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/triton_utils.hpp>

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

  TRITONSERVER_Error* LoadModel(
      std::string artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id);
  TRITONSERVER_Error* UnloadModel();

  // Get the handle to the TRITONBACKEND model.
  // TODO: Move to src
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  ML::fil::treelite_params_t tl_params;
  void* treelite_handle;

  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version,
      common::TritonJson::Value* model_config);

 private:
  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value * model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;
};

}}}
