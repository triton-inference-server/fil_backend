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

#include <cuml/fil/fil.h>
#include <raft/handle.hpp>
#include <treelite/c_api.h>

#include <limits>
#include <memory>
#include <thread>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

#include <triton_fil/enum_conversions.h>
#include <triton_fil/config.h>
#include <triton_fil/exceptions.h>
#include <triton_fil/model_state.h>
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_utils.h>
#include <triton_fil/triton_tensor.cuh>
#include <triton_fil/triton_tensor_utils.cuh>

namespace triton { namespace backend { namespace fil {

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  try {
    std::string name = get_backend_name(*backend);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

    if (!check_backend_version(*backend)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "triton backend API version does not support this backend");
    }
  } catch(TritonException& err) {
    return err.error();
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  try {
    std::string name = get_model_name(*model);

    uint64_t version = get_model_version(*model);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
         std::to_string(version) + ")")
            .c_str());

    set_model_state(*model, ModelState::Create(*model));

  } catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  try {
    auto model_state = get_model_state<ModelState>(*model);
    if (model_state != nullptr) {
      model_state->UnloadModel();
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

    delete model_state;
  } catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  try {
    std::string name = get_model_instance_name(*instance);
    int32_t device_id = get_device_id(*instance);
    TRITONSERVER_InstanceGroupKind kind = get_instance_kind(*instance);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
         TRITONSERVER_InstanceGroupKindString(kind) + " device " +
         std::to_string(device_id) + ")")
            .c_str());

    ModelState* model_state = get_model_state<ModelState>(*instance);

    set_instance_state<ModelInstanceState>(*instance,
                                           ModelInstanceState::Create(model_state,
                                                                      instance));

  } catch (TritonException& err) {
    return err.error();
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  try {
    void* vstate;
    triton_check(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state =
        reinterpret_cast<ModelInstanceState*>(vstate);

    instance_state->UnloadFILModel();

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

    delete instance_state;
  } catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** raw_requests,
    const uint32_t request_count)
{
  try {
    auto instance_state = get_instance_state<ModelInstanceState>(*instance);
    ModelState* model_state = instance_state->StateForModel();

    /* LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("model ") + model_state->Name() + ", instance " +
         instance_state->Name() + ", executing " + std::to_string(request_count) +
         " requests")
            .c_str()); */

    std::vector<TRITONBACKEND_Request*> requests(
      raw_requests, raw_requests + request_count
    );
    std::vector<TRITONBACKEND_Response*> responses = construct_responses(
      requests, request_count
    );

    try {
      auto input_batch = get_input_batch<float>(
        static_cast<uint32_t>(0),
        requests,
        TRITONSERVER_MEMORY_GPU,
        instance_state->get_raft_handle()
      );

      std::vector<int64_t> output_shape{input_batch.shape()[0]};
      if (model_state->predict_proba) {
        output_shape.push_back(model_state->num_class());
      }
      auto output_batch = get_output_batch<float>(
        static_cast<uint32_t>(0),
        requests,
        responses,
        TRITONSERVER_MEMORY_GPU,
        output_shape,
        instance_state->get_raft_handle()
      );

      instance_state->predict(
        input_batch,
        output_batch,
        model_state->predict_proba
      );

      output_batch.sync();
    } catch (TritonException& request_err) {
      std::fill(responses.begin(), responses.end(), nullptr);
      TRITONSERVER_ErrorDelete(request_err.error());
    }

    send_responses(responses);
    release_requests(requests);

  } catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::fil
