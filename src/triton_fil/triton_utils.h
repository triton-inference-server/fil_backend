/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace fil {

// TODO: Update to accept std::string for all log functions

/** Log message at indicated level */
void log(
    TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* message);


/** Log message at INFO level */
void log_info(const char* filename, const int line, const char* message);


/** Log message at WARN level */
void log_warn(const char* filename, const int line, const char* message);


/** Log message at ERROR level */
void log_error(const char* filename, const int line, const char* message);


/** Log message at VERBOSE level */
void log_debug(const char* filename, const int line, const char* message);


/** Get the name of the given backend */
std::string get_backend_name(TRITONBACKEND_Backend& backend);

/** Check if the backend version API that this backend was compiled against is
 * supported by Triton
 */
bool check_backend_version(TRITONBACKEND_Backend& backend);

/** Get the name of the given model */
std::string get_model_name(TRITONBACKEND_Model& model);

/** Get the version of the given model */
uint64_t get_model_version(TRITONBACKEND_Model& model);

/** Get JSON configuration for given model */
std::unique_ptr<common::TritonJson::Value> get_model_config(
    TRITONBACKEND_Model& model);

/** Get Triton server object for given model */
TRITONSERVER_Server* get_server(TRITONBACKEND_Model& model);

/** Set model's state to given object */
template <typename ModelStateType>
void
set_model_state(
    TRITONBACKEND_Model& model, std::unique_ptr<ModelStateType>&& model_state)
{
  triton_check(TRITONBACKEND_ModelSetState(
      &model, reinterpret_cast<void*>(model_state.release())));
}

/** Get state of given model */
template <typename ModelStateType>
ModelStateType*
get_model_state(TRITONBACKEND_Model& model)
{
  void* vstate;
  triton_check(TRITONBACKEND_ModelState(&model, &vstate));

  ModelStateType* model_state = reinterpret_cast<ModelStateType*>(vstate);

  return model_state;
}

/** Get the memory type corresponding to the instance type */
TRITONSERVER_MemoryType get_native_memory_for_instance(
    TRITONSERVER_InstanceGroupKind kind);

/** Get the name of the given model instance */
std::string get_model_instance_name(TRITONBACKEND_ModelInstance& instance);

/** Get the device_id for the given model instance */
int32_t get_device_id(TRITONBACKEND_ModelInstance& instance);

/** Get the group kind for the given model instance */
TRITONSERVER_InstanceGroupKind get_instance_kind(
    TRITONBACKEND_ModelInstance& instance);

/** Get the model associated with an instance */
TRITONBACKEND_Model* get_model_from_instance(
    TRITONBACKEND_ModelInstance& instance);

/** Get model state from instance */
template <typename ModelStateType>
ModelStateType*
get_model_state(TRITONBACKEND_ModelInstance& instance)
{
  return get_model_state<ModelStateType>(*get_model_from_instance(instance));
}

/** Set model instance state to given object */
template <typename ModelInstanceStateType>
void
set_instance_state(
    TRITONBACKEND_ModelInstance& instance,
    std::unique_ptr<ModelInstanceStateType>&& model_instance_state)
{
  triton_check(TRITONBACKEND_ModelInstanceSetState(
      &instance, reinterpret_cast<void*>(model_instance_state.release())));
}

/** Get model instance state from instance */
template <typename ModelInstanceStateType>
ModelInstanceStateType*
get_instance_state(TRITONBACKEND_ModelInstance& instance)
{
  ModelInstanceStateType* instance_state;
  triton_check(TRITONBACKEND_ModelInstanceState(
      &instance, reinterpret_cast<void**>(&instance_state)));
  return instance_state;
}

/** Construct empty response objects for given requests */
template <class Iter>
typename std::enable_if_t<
    std::is_same<
        typename std::iterator_traits<Iter>::value_type,
        TRITONBACKEND_Request*>::value,
    std::vector<TRITONBACKEND_Response*>>
construct_responses(Iter requests_begin, Iter requests_end)
{
  std::vector<TRITONBACKEND_Response*> responses;
  auto requests_size = std::distance(requests_begin, requests_end);
  assert(requests_size > 0);
  responses.reserve(requests_size);
  std::transform(
      requests_begin, requests_end, std::back_inserter(responses),
      [](TRITONBACKEND_Request* request) {
        TRITONBACKEND_Response* response;
        triton_check(TRITONBACKEND_ResponseNew(&response, request));
        return response;
      });
  return responses;
}

/** Send responses */
void send_responses(
    std::vector<TRITONBACKEND_Response*>& responses,
    TRITONSERVER_Error* err = nullptr);

/** Convenience method for sending error responses for a batch of requests
 * whose responses have not yet been constructed */
template <class Iter>
typename std::enable_if_t<std::is_same<
    typename std::iterator_traits<Iter>::value_type,
    TRITONBACKEND_Request*>::value>
send_error_responses(
    Iter requests_begin, Iter requests_end, TRITONSERVER_Error* err)
{
  auto responses = construct_responses(requests_begin, requests_end);
  send_responses(responses, err);
}

/** Release requests */
template <class Iter>
typename std::enable_if_t<std::is_same<
    typename std::iterator_traits<Iter>::value_type,
    TRITONBACKEND_Request*>::value>
release_requests(Iter begin, Iter end)
{
  std::for_each(begin, end, [](decltype(*begin) request) {
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  });
}

/** Report statistics for a group of requests
 *
 * @param instance The model instance which processed requests
 * @param requests_begin Iterator to first processed request
 * @param requests_end Iterator one past last processed request
 * @param success Boolean indicating whether these requests were successfully
 * processed
 * @param start_time Timestamp in nanoseconds since the epoch at which
 * processing of any kind first began on these requests
 * @param compute_start_time Timestamp in nanoseconds since the epoch at which
 * actual inference computations began on these requests
 * @param compute_end_time Timestamp in nanoseconds since the epoch at which
 * actual inference computations began on these requests
 * @param end_time Timestamp in nanoseconds since the epoch at which last
 * processing of any kind occurred on these requests
 */
template <class Iter>
typename std::enable_if_t<std::is_same<
    typename std::iterator_traits<Iter>::value_type,
    TRITONBACKEND_Request*>::value>
report_statistics(
    TRITONBACKEND_ModelInstance& instance, Iter requests_begin,
    Iter requests_end, bool success, uint64_t start_time,
    uint64_t compute_start_time, uint64_t compute_end_time, uint64_t end_time)
{
  std::for_each(
      requests_begin, requests_end, [&](TRITONBACKEND_Request* request) {
        auto err = TRITONBACKEND_ModelInstanceReportStatistics(
            &instance, request, success, start_time, compute_start_time,
            compute_end_time, end_time);
        if (err != nullptr) {
          throw(TritonException(err));
        }
      });
}

/** Report statistics for a Triton-defined batch of requests
 *
 * @param instance The model instance which processed requests
 * @param requests A vector of TRITONBACKEND_Request pointers which have been
 * processed
 * @param success Boolean indicating whether these requests were successfully
 * processed
 * @param start_time Timestamp in nanoseconds since the epoch at which
 * processing of any kind first began on these requests
 * @param compute_start_time Timestamp in nanoseconds since the epoch at which
 * actual inference computations began on these requests
 * @param compute_end_time Timestamp in nanoseconds since the epoch at which
 * actual inference computations began on these requests
 * @param end_time Timestamp in nanoseconds since the epoch at which last
 * processing of any kind occurred on these requests
 */
void report_statistics(
    TRITONBACKEND_ModelInstance& instance, std::size_t inference_count,
    uint64_t start_time, uint64_t compute_start_time, uint64_t compute_end_time,
    uint64_t end_time);
}}}  // namespace triton::backend::fil
