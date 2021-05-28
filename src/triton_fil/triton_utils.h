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

#include <memory>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace fil {

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
std::vector<TRITONBACKEND_Response*> construct_responses(
    std::vector<TRITONBACKEND_Request*>& requests);

/** Send responses */
void send_responses(
    std::vector<TRITONBACKEND_Response*>& responses,
    TRITONSERVER_Error* err = nullptr);

/** Convenience method for sending error responses for a batch of requests
 * whose responses have not yet been constructed */
void send_error_responses(
    std::vector<TRITONBACKEND_Request*>& requests, TRITONSERVER_Error* err);

/** Release requests */
void release_requests(std::vector<TRITONBACKEND_Request*>& requests);

}}}  // namespace triton::backend::fil
