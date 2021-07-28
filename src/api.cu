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
#include <treelite/c_api.h>
#include <raft/handle.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

#include <triton_fil/config.h>
#include <triton_fil/enum_conversions.h>
#include <triton_fil/exceptions.h>
#include <triton_fil/model_state.h>
#include <triton_fil/triton_utils.h>
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_tensor.cuh>
#include <triton_fil/triton_tensor_utils.cuh>

namespace triton { namespace backend { namespace fil {

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  try {
    std::string name = get_backend_name(*backend);

    log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

    if (!check_backend_version(*backend)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "triton backend API version does not support this backend");
    }
  }
  catch (TritonException& err) {
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

    log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
         std::to_string(version) + ")")
            .c_str());

    set_model_state(*model, ModelState::Create(*model));
  }
  catch (TritonException& err) {
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

    log_info(
        __FILE__, __LINE__, "TRITONBACKEND_ModelFinalize: delete model state");

    delete model_state;
  }
  catch (TritonException& err) {
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

    log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
         TRITONSERVER_InstanceGroupKindString(kind) + " device " +
         std::to_string(device_id) + ")")
            .c_str());

    ModelState* model_state = get_model_state<ModelState>(*instance);

    set_instance_state<ModelInstanceState>(
        *instance, ModelInstanceState::Create(model_state, instance));
  }
  catch (TritonException& err) {
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

    if (instance_state != nullptr) {
      instance_state->UnloadFILModel();

      log_info(
          __FILE__, __LINE__,
          "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

      delete instance_state;
    }
  }
  catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** raw_requests,
    const uint32_t request_count)
{
  uint64_t all_start_time =
      std::chrono::steady_clock::now().time_since_epoch().count();

  std::vector<TRITONBACKEND_Request*> requests(
      raw_requests, raw_requests + request_count);

  auto instance_state = get_instance_state<ModelInstanceState>(*instance);
  auto model_state = instance_state->StateForModel();
  auto target_memory = get_native_memory_for_instance(instance_state->Kind());

  // Get all input data and bail out gracefully if this fails
  std::vector<InputBatch<float>> input_batches;
  try {
    input_batches = get_input_batches<float>(
        static_cast<uint32_t>(0), requests, target_memory,
        instance_state->get_raft_handle());
  } catch (TritonException& setup_err) {
    return setup_err.error();
  }

  // BEGIN: Lambda definitions for processing individual batches

  /* Given an input batch and output batch, perform prediction on input,
   * storing results in output and returning timing information. Return value
   * is a std::optional containing a pair of uint64_t representing nanoseconds
   * since the epoch to the beginning and end of prediction respectively. If
   * std::optional is std::nullopt, prediction has failed and timing values
   * would have no meaning */
  auto instance_predict = [&](
      decltype(*input_batches.begin())& input_batch,
      TritonTensor<float>& output_batch
  ){
    uint64_t start_time =
        std::chrono::steady_clock::now().time_since_epoch().count();
    instance_state->predict(
        input_batch.data, output_batch, model_state->predict_proba);
    uint64_t end_time =
        std::chrono::steady_clock::now().time_since_epoch().count();
    output_batch.sync();
    return std::pair<uint64_t, uint64_t>{start_time, end_time};
  };


  /* Given a vector of input batches, return a vector of vectors representing
   * the output shape of each batch */
  auto get_output_shapes = [&](decltype(*input_batches.begin())& batch) {
    std::vector<std::vector<int64_t>> output_shapes;
    output_shapes.reserve(batch.shapes.size());
    std::transform(
      batch.shapes.begin(),
      batch.shapes.end(),
      std::back_inserter(output_shapes),
      [&] (decltype(*batch.shapes.begin())& input_shape) { return
      model_state->get_output_shape(input_shape);}
    );
    return output_shapes;
  };

  /* Given start and end iterators for requests, process them and send
   * responses. Return a std::optional containing a pair of values representing
   * the time in nanoseconds since the epoch at which computation began and
   * ended for this batch.
   *
   * This lambda is guaranteed to send a response for each request in the batch
   */
  auto respond_to_requests = [&](decltype(*input_batches.begin())& batch, decltype(requests.begin()) requests_begin,
      decltype(requests.begin()) requests_end) {
      auto responses = construct_responses(requests_begin, requests_end);
      // TODO: Avoid constructing this vector; pass iterators instead
      std::vector<std::remove_reference<decltype(*requests_begin)>::type>
        batch_requests(requests_begin, requests_end);
      std::optional<std::pair<uint64_t, uint64_t>> timings{std::nullopt};

      try {
        auto output_batch = get_output_batch<float>(
            static_cast<uint32_t>(0), batch_requests, responses,
            target_memory, get_output_shapes(batch), instance_state->get_raft_handle());

        timings = instance_predict(batch, output_batch);

        send_responses(responses);
      } catch (TritonException& predict_err) {
        // Respond with error message on failure
        send_responses(responses, predict_err.error());
      }

      return timings;
  };

  /* Perform all processing for a single batch, add the number of samples to
   * the running total of processed samples, and return the new total.
   * Processing includes prediction, sending responses, reporting statistics,
   * and releasing requests. This lambda is guaranteed to send some response
   * for each request, report statistics, and release all requests in the batch
   * regardless of other failures */
  auto process_batch = [&](std::size_t processed_count, decltype(*input_batches.begin())& batch) {
    uint64_t batch_start_time =
        std::chrono::steady_clock::now().time_since_epoch().count();

    auto batch_requests_begin = requests.begin() + batch.extent.first;
    auto batch_requests_end = requests.begin() + batch.extent.second;

    decltype(get_output_shapes(batch)) output_shapes;
    try {
      output_shapes = get_output_shapes(batch);
    } catch (TritonException& setup_err) {
      // Always report statistics, send *some* response, and release
      // requests for this batch
      uint64_t now =
        std::chrono::steady_clock::now().time_since_epoch().count();
      instance_state->report_statistics(
          batch_requests_begin, batch_requests_end, false, batch_start_time, now, now, now);
      send_error_responses(batch_requests_begin, batch_requests_end, setup_err.error());
      release_requests(batch_requests_begin, batch_requests_end);
      return processed_count;
    }

    auto timings = respond_to_requests(batch, batch_requests_begin, batch_requests_end);

    auto batch_compute_start_time = uint64_t{0};
    auto batch_compute_end_time = uint64_t{0};

    if (timings) {
      batch_compute_start_time = timings->first;
      batch_compute_end_time = timings->second;
      processed_count = std::accumulate(
        output_shapes.begin(), output_shapes.end(), processed_count,
        [](std::size_t sum, decltype(*output_shapes.begin())& shape) {
          return sum + *shape.begin();
        }
      );
    }

    instance_state->report_statistics(
        batch_requests_begin, batch_requests_end, timings.has_value(), batch_start_time,
        batch_compute_start_time, batch_compute_end_time,
        std::chrono::steady_clock::now().time_since_epoch().count());

    release_requests(batch_requests_begin, batch_requests_end);

    return processed_count;
  };

  // END: Lambda definitions for processing individual batches

  auto total_inference_count = std::accumulate(input_batches.begin(),
      input_batches.end(), std::size_t{0}, process_batch);

  uint64_t all_end_time =
      std::chrono::steady_clock::now().time_since_epoch().count();

  instance_state->report_statistics(
    total_inference_count, all_start_time, all_start_time,
    all_end_time, all_end_time);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::fil
