#include <triton/backend/backend_common.h>

#include <triton_fil/c_wrappers.hpp>
#include <triton_fil/model_state.hpp>

namespace triton { namespace backend { namespace fil {

extern "C" {

TRITONSERVER_Error*
unload_treelite_model(ModelState* state)
{
  state->UnloadModel();
  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::fil
