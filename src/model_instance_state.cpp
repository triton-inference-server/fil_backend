#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>

#include <triton_fil/c_wrappers.hpp>
#include <triton_fil/model_instance_state.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id);
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::predict(
    const float* data, float* preds, size_t num_rows, bool predict_proba)
{
  ML::fil::predict(*handle, fil_forest, preds, data, num_rows, predict_proba);
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::to_device(
    float*& buffer_d, const float* buffer_h, size_t size)
{
  raft::allocate(buffer_d, size * sizeof(float));
  raft::copy(buffer_d, buffer_h, size, handle->get_stream());
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::to_host(
    float*& buffer_h, const float* buffer_d, size_t size)
{
  raft::copy(buffer_h, buffer_d, size, handle->get_stream());
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
