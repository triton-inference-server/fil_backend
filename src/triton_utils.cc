// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>
#include <triton_fil/triton_utils.h>

#include <chrono>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace fil {

void
log(TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* message)
{
  triton_check(TRITONSERVER_LogMessage(level, filename, line, message));
}

void
log_info(const char* filename, const int line, const char* message)
{
  log(TRITONSERVER_LOG_INFO, filename, line, message);
}

void
log_warn(const char* filename, const int line, const char* message)
{
  log(TRITONSERVER_LOG_WARN, filename, line, message);
}

void
log_error(const char* filename, const int line, const char* message)
{
  log(TRITONSERVER_LOG_ERROR, filename, line, message);
}

void
log_debug(const char* filename, const int line, const char* message)
{
  log(TRITONSERVER_LOG_VERBOSE, filename, line, message);
}

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

TRITONSERVER_MemoryType get_native_memory_for_instance(
    TRITONSERVER_InstanceGroupKind kind) {
  switch (kind) {
   case TRITONSERVER_INSTANCEGROUPKIND_GPU:
    return TRITONSERVER_MEMORY_GPU;
   case TRITONSERVER_INSTANCEGROUPKIND_CPU:
    return TRITONSERVER_MEMORY_CPU;
   default:
    throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        "Instance kind must be set to either GPU or CPU");
    return TRITONSERVER_MEMORY_CPU;
  }
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
construct_responses(std::vector<TRITONBACKEND_Request*>& requests)
{
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(requests.size());

  for (auto& request : requests) {
    TRITONBACKEND_Response* response;
    triton_check(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }
  return responses;
}

void
send_responses(
    std::vector<TRITONBACKEND_Response*>& responses, TRITONSERVER_Error* err)
{
  for (auto& response : responses) {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");
  }
}

void
send_error_responses(
    std::vector<TRITONBACKEND_Request*>& requests, TRITONSERVER_Error* err)
{
  auto responses = construct_responses(requests);
  send_responses(responses, err);
}

void
release_requests(std::vector<TRITONBACKEND_Request*>& requests)
{
  for (auto& request : requests) {
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
}

void
report_statistics(
    TRITONBACKEND_ModelInstance& instance,
    std::vector<TRITONBACKEND_Request*>& requests, bool success,
    uint64_t start_time, uint64_t compute_start_time, uint64_t compute_end_time,
    uint64_t end_time)
{
  for (auto request : requests) {
    auto err = TRITONBACKEND_ModelInstanceReportStatistics(
        &instance, request, success, start_time, compute_start_time,
        compute_end_time, end_time);
    if (err != nullptr) {
      throw(TritonException(err));
    }
  }
}

void
report_statistics(
    TRITONBACKEND_ModelInstance& instance, std::size_t inference_count,
    uint64_t start_time, uint64_t compute_start_time, uint64_t compute_end_time,
    uint64_t end_time)
{
  auto err = TRITONBACKEND_ModelInstanceReportBatchStatistics(
      &instance, inference_count, start_time, compute_start_time,
      compute_end_time, end_time);
  if (err != nullptr) {
    throw(TritonException(err));
  }
}

}}}  // namespace triton::backend::fil
