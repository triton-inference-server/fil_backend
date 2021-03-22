#pragma once

#include <triton/backend/backend_common.h>

namespace triton { namespace backend { namespace fil {

class ModelState;

extern "C" {

TRITONSERVER_Error* unload_treelite_model(ModelState* state);

}  // extern "C"

}}}  // namespace triton::backend::fil
