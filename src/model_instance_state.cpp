#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>

#include <memory>
#include <raft/handle.hpp>
#include <triton_fil/model_instance_state.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

std::unique_ptr<ModelInstanceState>
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
{
  std::string instance_name = get_model_instance_name(*triton_model_instance);
  TRITONSERVER_InstanceGroupKind instance_kind =
      get_instance_kind(*triton_model_instance);
  int32_t instance_id = get_device_id(*triton_model_instance);

  return std::make_unique<ModelInstanceState>(
      model_state, triton_model_instance, instance_name.c_str(), instance_kind,
      instance_id);
}

raft::handle_t&
ModelInstanceState::get_raft_handle()
{
  return *handle;
}

TRITONSERVER_Error*
ModelInstanceState::predict(
    const float* data, float* preds, size_t num_rows, bool predict_proba)
{
  ML::fil::predict(*handle, fil_forest, preds, data, num_rows, predict_proba);
  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id),
      handle(std::make_unique<raft::handle_t>())
{
  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId()));
  ML::fil::from_treelite(
      *handle, &fil_forest, model_state_->treelite_handle,
      &(model_state_->tl_params));
}

void
ModelInstanceState::UnloadFILModel()
{
  ML::fil::free(*handle, fil_forest);
}

}}}  // namespace triton::backend::fil
