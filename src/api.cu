// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cuml/fil/fil.h>
#include <raft/handle.hpp>
#include <treelite/c_api.h>

#include <limits>
#include <memory>
#include <thread>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

#include <triton_fil/c_wrappers.hpp>
#include <triton_fil/enum_conversions.hpp>
#include <triton_fil/config.hpp>
#include <triton_fil/exceptions.hpp>
#include <triton_fil/macros.h>
#include <triton_fil/model_state.hpp>
#include <triton_fil/model_instance_state.hpp>
#include <triton_fil/triton_utils.hpp>

namespace triton { namespace backend { namespace fil {

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
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

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
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
  } catch (TritonException& err) {
    return err.error();
  }

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  auto model_state = ModelState::Create(*model);
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model,
        reinterpret_cast<void*>(model_state.release())));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  RETURN_IF_ERROR(unload_treelite_model(model_state));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // Because this backend just copies IN -> OUT and requires that
  // input and output be in CPU memory, we fail if a GPU instances is
  // requested.
  /* RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'identity' backend only supports CPU instances")); */
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  unload_fil_model(instance_state);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  for (uint32_t r = 0; r < request_count; ++r) {

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name, &input));

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputName(
            request, 0 /* index */, &requested_output_name));

    // If an error response was sent while getting the input or
    // requested output name then display an error message and move on
    // to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input or requested output name, error response "
           "sent")
              .c_str());
      continue;
    }

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\tinput ") + input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
         ", byte_size = " + std::to_string(input_byte_size) +
         ", buffer_count = " + std::to_string(input_buffer_count))
            .c_str());
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\trequested_output ") + requested_output_name).c_str());

    // For statistics we need to collect the total batch size of all
    // the requests. If the model doesn't support batching then each
    // request is necessarily batch-size 1. If the model does support
    // batching then the first dimension of the shape is the batch
    // size.
    if (supports_batching && (input_dims_count > 0)) {
      total_batch_size += input_shape[0];
    } else {
      total_batch_size++;
    }

    // This backend simply copies the input tensor to the output
    // tensor. The input tensor contents are available in one or
    // more contiguous buffers. To do the copy we:
    //
    //   1. Create an output tensor in the response.
    //
    //   2. Allocate appropriately sized buffer in the output
    //      tensor.
    //
    //   3. Iterate over the input tensor buffers and copy the
    //      contents into the output buffer.
    TRITONBACKEND_Response* response = responses[r];

    // Step 1 
    TRITONBACKEND_Output* output;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_ResponseOutput(
            response, &output, requested_output_name, input_datatype,
            input_shape, input_dims_count - 1));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create response output, error response sent")
              .c_str());
      continue;
    }

    // Step 2. Get the output buffer. We request a buffer in CPU
    // memory but we have to handle any returned type. If we get
    // back a buffer in GPU memory we just fail the request.
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_OutputBuffer(
            output,
            &output_buffer,
            input_byte_size / input_shape[input_dims_count - 1],
            &output_memory_type,
            &output_memory_type_id));
    if ((responses[r] == nullptr) ||
        (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to create output buffer in CPU memory"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create output buffer in CPU memory, error response "
           "sent")
              .c_str());
      continue;
    }
    float* output_buffer_device = nullptr;
    fil_to_device(
      instance,
      output_buffer_device,
      reinterpret_cast<const float*>(output_buffer),
      input_byte_size / input_shape[input_dims_count - 1] / sizeof(float)
    );

    // Step 3. Copy input -> output. We can only handle if the input
    // buffers are on CPU so fail otherwise.
    size_t output_buffer_offset = 0;
    for (uint32_t b = 0; b < input_buffer_count; ++b) {
      const void* input_buffer = nullptr;
      uint64_t buffer_byte_size = 0;
      TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t input_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputBuffer(
              input, b, &input_buffer, &buffer_byte_size, &input_memory_type,
              &input_memory_type_id));

      if ((responses[r] == nullptr) ||
          (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to get input buffer in CPU memory"));
      }

      float* input_buffer_device = nullptr;
      fil_to_device(
        instance,
        input_buffer_device,
        reinterpret_cast<const float*>(input_buffer),
        buffer_byte_size / sizeof(float)
      );

      fil_predict(
        instance,
        input_buffer_device,
        output_buffer_device + output_buffer_offset,
        static_cast<size_t>(input_shape[0])
      );
      output_buffer_offset += buffer_byte_size;
      CUDA_CHECK(cudaFree(input_buffer_device));
    }

    float * output_buffer_float = reinterpret_cast<float*>(output_buffer);

    fil_to_host(
      instance,
      output_buffer_float,
      output_buffer_device,
      input_byte_size / input_shape[input_dims_count - 1] / sizeof(float)
    );

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to get input buffer in CPU memory, error response "
           "sent")
              .c_str());
      continue;
    }

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
    CUDA_CHECK(cudaFree(output_buffer_device));
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::fil
