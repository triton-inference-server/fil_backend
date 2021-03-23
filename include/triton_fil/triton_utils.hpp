#pragma once
#include <triton/backend/backend_common.h>

namespace triton { namespace backend { namespace fil {

/** Get the name of the given backend */
std::string get_backend_name(TRITONBACKEND_Backend& backend);

/** Check if the backend version API that this backend was compiled against is
 * supported by Triton
 */
bool check_backend_version(TRITONBACKEND_Backend& backend);

}}}
