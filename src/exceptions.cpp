#include <triton/core/tritonserver.h>

#include <triton_fil/exceptions.hpp>

namespace triton { namespace backend { namespace fil {

void
triton_check(TRITONSERVER_Error* err)
{
  if (err != nullptr) {
    throw TritonException(err);
  }
}

}}}  // namespace triton::backend::fil
