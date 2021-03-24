#include <triton/core/tritonbackend.h>
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

struct backend_version {
  uint32_t major;
  uint32_t minor;
};

bool
check_backend_version(TRITONBACKEND_Backend& backend)
{
  backend_version version;
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

uint64_t
get_model_version(TRITONBACKEND_Model& model)
{
  uint64_t version;
  TRITONSERVER_Error* result;
  result = TRITONBACKEND_ModelVersion(&model, &version);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return version;
}

std::string
get_model_name(TRITONBACKEND_Model& model)
{
  const char* cname;
  TRITONSERVER_Error* result = TRITONBACKEND_ModelName(&model, &cname);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return std::string(cname);
}

std::unique_ptr<common::TritonJson::Value>
get_model_config(TRITONBACKEND_Model& model)
{
  TRITONSERVER_Message* config_message;
  TRITONSERVER_Error* result =
      TRITONBACKEND_ModelConfig(&model, 1, &config_message);
  if (result != nullptr) {
    throw(TritonException(result));
  }

  const char* buffer;
  size_t byte_size;
  result =
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size);
  if (result != nullptr) {
    throw(TritonException(result));
  }

  auto model_config = std::make_unique<common::TritonJson::Value>();
  TRITONSERVER_Error* err = model_config->Parse(buffer, byte_size);
  result = TRITONSERVER_MessageDelete(config_message);
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
  TRITONSERVER_Error* result = TRITONBACKEND_ModelServer(&model, &server);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return server;
}

std::string
get_model_instance_name(TRITONBACKEND_ModelInstance& instance)
{
  const char* cname;
  TRITONSERVER_Error* result =
      TRITONBACKEND_ModelInstanceName(&instance, &cname);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return std::string(cname);
}

int32_t
get_device_id(TRITONBACKEND_ModelInstance& instance)
{
  int32_t device_id;
  TRITONSERVER_Error* result =
      TRITONBACKEND_ModelInstanceDeviceId(&instance, &device_id);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return device_id;
}

TRITONSERVER_InstanceGroupKind
get_instance_kind(TRITONBACKEND_ModelInstance& instance)
{
  TRITONSERVER_InstanceGroupKind kind;
  TRITONSERVER_Error* result =
      TRITONBACKEND_ModelInstanceKind(&instance, &kind);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return kind;
}

TRITONBACKEND_Model*
get_model_from_instance(TRITONBACKEND_ModelInstance& instance)
{
  TRITONBACKEND_Model* model;
  TRITONSERVER_Error* result =
      TRITONBACKEND_ModelInstanceModel(&instance, &model);
  if (result != nullptr) {
    throw(TritonException(result));
  }
  return model;
}

}}}  // namespace triton::backend::fil
