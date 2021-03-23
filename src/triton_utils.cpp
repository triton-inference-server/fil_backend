#include <triton/core/tritonserver.h>

#include <string>
#include <triton_fil/exceptions.hpp>
#include <triton_fil/triton_utils.hpp>

namespace triton { namespace backend { namespace fil {

std::string
get_backend_name(TRITONBACKEND_Backend& backend)
{
  const char* cname;
  TRITONSERVER_Error* result = TRITONBACKEND_BackendName(&backend, &cname);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return std::string(cname);
}

struct triton_version {
  uint32_t major;
  uint32_t minor;
};

bool
check_backend_version(TRITONBACKEND_Backend& backend)
{
  triton_version version;
  TRITONSERVER_Error* result;
  result = TRITONBACKEND_ApiVersion(&version.major, &version.minor);
  if (result != nullptr) {
    throw(TritonException(result));
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(version.major) + "." + std::to_string(version.minor))
          .c_str());

  std::string name = get_backend_name(backend);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  return (
      (version.major == TRITONBACKEND_API_VERSION_MAJOR) &&
      (version.minor >= TRITONBACKEND_API_VERSION_MINOR));
}

}}}  // namespace triton::backend::fil
