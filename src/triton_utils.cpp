#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>

#include <string>
#include <triton_fil/exceptions.hpp>
#include <triton_fil/triton_utils.hpp>
#include <vector>

namespace triton { namespace backend { namespace fil {

std::string
get_backend_name(TRITONBACKEND_Backend& backend)
{
  const char* cname;
  triton_check(TRITONBACKEND_BackendName(&backend, &cname));
  return std::string(cname);
}

struct backend_version {
  uint32_t major;
  uint32_t minor;
};

bool
check_backend_version(TRITONBACKEND_Backend& backend)
{
  backend_version version;
  triton_check(TRITONBACKEND_ApiVersion(&version.major, &version.minor));

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

uint64_t
get_model_version(TRITONBACKEND_Model& model)
{
  uint64_t version;
  triton_check(TRITONBACKEND_ModelVersion(&model, &version));
  return version;
}

std::string
get_model_name(TRITONBACKEND_Model& model)
{
  const char* cname;
  triton_check(TRITONBACKEND_ModelName(&model, &cname));
  return std::string(cname);
}

std::unique_ptr<common::TritonJson::Value>
get_model_config(TRITONBACKEND_Model& model)
{
  TRITONSERVER_Message* config_message;
  triton_check(TRITONBACKEND_ModelConfig(&model, 1, &config_message));

  const char* buffer;
  size_t byte_size;
  triton_check(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  auto model_config = std::make_unique<common::TritonJson::Value>();
  TRITONSERVER_Error* err = model_config->Parse(buffer, byte_size);
  TRITONSERVER_Error* result = TRITONSERVER_MessageDelete(config_message);
  if (err != nullptr) {
    throw(TritonException(err));
  }
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return model_config;
}

TRITONSERVER_Server*
get_server(TRITONBACKEND_Model& model)
{
  TRITONSERVER_Server* server;
  triton_check(TRITONBACKEND_ModelServer(&model, &server));
  return server;
}

std::string
get_model_instance_name(TRITONBACKEND_ModelInstance& instance)
{
  const char* cname;
  triton_check(TRITONBACKEND_ModelInstanceName(&instance, &cname));
  return std::string(cname);
}

int32_t
get_device_id(TRITONBACKEND_ModelInstance& instance)
{
  int32_t device_id;
  triton_check(TRITONBACKEND_ModelInstanceDeviceId(&instance, &device_id));
  return device_id;
}

TRITONSERVER_InstanceGroupKind
get_instance_kind(TRITONBACKEND_ModelInstance& instance)
{
  TRITONSERVER_InstanceGroupKind kind;
  triton_check(TRITONBACKEND_ModelInstanceKind(&instance, &kind));
  return kind;
}

TRITONBACKEND_Model*
get_model_from_instance(TRITONBACKEND_ModelInstance& instance)
{
  TRITONBACKEND_Model* model;
  triton_check(TRITONBACKEND_ModelInstanceModel(&instance, &model));
  return model;
}

std::vector<TRITONBACKEND_Response*>
construct_responses(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    triton_check(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }
  return responses;
}

}}}  // namespace triton::backend::fil
