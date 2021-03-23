#include <triton/backend/backend_common.h>

#include <triton_fil/c_wrappers.hpp>
#include <triton_fil/model_instance_state.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

extern "C" {

TRITONSERVER_Error*
unload_treelite_model(ModelState* state)
{
  state->UnloadModel();
  return nullptr;
}

TRITONSERVER_Error*
fil_predict(
    TRITONBACKEND_ModelInstance* instance, const float* input_buffer,
    float* output_buffer, size_t rows)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  instance_state->predict(input_buffer, output_buffer, rows);

  return nullptr;
}

extern "C" TRITONSERVER_Error*
unload_fil_model(ModelInstanceState* instance)
{
  instance->UnloadFILModel();
  return nullptr;
}

extern "C" TRITONSERVER_Error*
fil_to_device(
    TRITONBACKEND_ModelInstance* instance, float*& buffer_d,
    const float* buffer_h, size_t size)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  instance_state->to_device(buffer_d, buffer_h, size);

  return nullptr;
}

extern "C" TRITONSERVER_Error*
fil_to_host(
    TRITONBACKEND_ModelInstance* instance, float*& buffer_h,
    const float* buffer_d, size_t size)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  instance_state->to_host(buffer_h, buffer_d, size);

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::fil
