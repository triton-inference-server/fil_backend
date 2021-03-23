#pragma once

#include <triton/backend/backend_common.h>

namespace triton { namespace backend { namespace fil {

class ModelState;
class ModelInstanceState;

extern "C" {

TRITONSERVER_Error* unload_treelite_model(ModelState* state);
TRITONSERVER_Error* fil_predict(TRITONBACKEND_ModelInstance* instance,
                                const float* input_buffer,
                                float* output_buffer,
                                size_t rows);
TRITONSERVER_Error* unload_fil_model(ModelInstanceState * instance);
TRITONSERVER_Error* fil_to_device(TRITONBACKEND_ModelInstance * instance,
                                  float* & buffer_d,
                                  const float* buffer_h,
                                  size_t size);
TRITONSERVER_Error* fil_to_host(TRITONBACKEND_ModelInstance * instance,
                                float* & buffer_h,
                                const float* buffer_d,
                                size_t size);

}  // extern "C"

}}}  // namespace triton::backend::fil
