#pragma once
#include <memory>
#include <string>

#include <cuml/fil/fil.h>
#include <raft/handle.hpp>
#include <triton_fil/model_state.hpp>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>

namespace triton { namespace backend { namespace fil {

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static std::unique_ptr<ModelInstanceState> Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  void UnloadFILModel();
  TRITONSERVER_Error * predict(
      const float* data,
      float* preds,
      size_t num_rows,
      bool predict_proba = false);

  raft::handle_t& get_raft_handle();

  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

 private:
  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  ML::fil::forest_t fil_forest;
  std::unique_ptr<raft::handle_t> handle;
};

}}}
