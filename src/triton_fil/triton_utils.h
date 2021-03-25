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
std::unique_ptr<common::TritonJson::Value> get_model_config(TRITONBACKEND_Model& model);

/** Get Triton server object for given model */
TRITONSERVER_Server* get_server(TRITONBACKEND_Model& model);

/** Set model's state to given object */
template <typename ModelStateType>
void set_model_state(TRITONBACKEND_Model& model,
                     std::unique_ptr<ModelStateType>&& model_state) {
  triton_check(TRITONBACKEND_ModelSetState(
    &model,
    reinterpret_cast<void*>(model_state.release())
  ));
}

/** Get state of given model */
template <typename ModelStateType>
ModelStateType* get_model_state(TRITONBACKEND_Model& model) {
  void* vstate;
  triton_check(TRITONBACKEND_ModelState(&model, &vstate));

  ModelStateType* model_state = reinterpret_cast<ModelStateType*>(vstate);

  return model_state;
}

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
ModelStateType* get_model_state(TRITONBACKEND_ModelInstance& instance) {
  return get_model_state<ModelStateType>(*get_model_from_instance(instance));
}

/** Set model instance state to given object */
template <typename ModelInstanceStateType>
void set_instance_state(TRITONBACKEND_ModelInstance& instance,
                        std::unique_ptr<ModelInstanceStateType>&& model_instance_state) {
  triton_check(TRITONBACKEND_ModelInstanceSetState(
    &instance,
    reinterpret_cast<void*>(model_instance_state.release())
  ));
}

/** Get model instance state from instance */
template <typename ModelInstanceStateType>
ModelInstanceStateType* get_instance_state(TRITONBACKEND_ModelInstance& instance) {
  ModelInstanceStateType* instance_state;
  triton_check(TRITONBACKEND_ModelInstanceState(
    &instance,
    reinterpret_cast<void**>(&instance_state)
  ));
  return instance_state;
}

/** Construct empty response objects for given requests */
std::vector<TRITONBACKEND_Response*> construct_responses(
  TRITONBACKEND_Request** requests,
  const uint32_t request_count);

}}}
