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

#include <raft/cudart_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/model_state.h>

#include <memory>
#include <raft/handle.hpp>
#include <treelite/c_api.h>
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_tensor.cuh>

namespace triton { namespace backend { namespace fil {

std::unique_ptr<ModelInstanceState>
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
{
  std::string instance_name = get_model_instance_name(*triton_model_instance);
  int32_t instance_id = get_device_id(*triton_model_instance);

  return std::make_unique<ModelInstanceState>(
      model_state, triton_model_instance, instance_name.c_str(), instance_id);
}

raft::handle_t*
ModelInstanceState::get_raft_handle()
{
  return handle.get();
}

void
ModelInstanceState::predict(
    TritonTensor<const float>& data, TritonTensor<float>& preds,
    bool predict_proba)
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    try {
      ML::fil::predict(
          *handle, fil_forest, preds.data(), data.data(), data.shape()[0],
          predict_proba);
    }
    catch (raft::cuda_error& err) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL, err.what());
    }
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    {
      std::ostringstream oss;
      oss << "preds.shape = ";
      for (int64_t e : preds.shape()) {
        oss << e << ", ";
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, oss.str().c_str());
    }
    {
      std::ostringstream oss;
      oss << "data.shape = ";
      for (int64_t e : data.shape()) {
        oss << e << ", ";
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, oss.str().c_str());
    }

    std::size_t output_size;
    int res = TreeliteGTILGetPredictOutputSize(model_state_->treelite_handle,
                  data.shape()[0], &output_size);
    if (res != 0) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          TreeliteGetLastError());
    }

    std::string msg = std::string("output_size = ") + std::to_string(output_size);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, msg.c_str());

    std::vector<float> out_buffer(data.shape()[0] * output_size);
    std::size_t out_result_size;
    {
      char buf[1000] = {0};
      sprintf(buf, "treelite_handle = %p", model_state_->treelite_handle);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, buf);
    }
    res = TreeliteGTILPredict(model_state_->treelite_handle, data.data(),
                data.shape()[0], out_buffer.data(), 1, &out_result_size);
    if (res != 0) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          TreeliteGetLastError());
    }

    msg = std::string("out_result_size = ") + std::to_string(out_result_size);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, msg.c_str());

    if (out_result_size > 1 && !predict_proba) {
      // Compute argmax and return the best class
      if (preds.size() != 1) {
        throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            "Failed assumption: preds was assumed to be 1-element long");
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
      float max_score = out_buffer[0];
      std::size_t best_class = 0;
      for (std::size_t i = 1; i < out_result_size; ++i) {
        if (out_buffer[i] > max_score) {
          max_score = out_buffer[i];
          best_class = i;
        }
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
      preds.data()[0] = static_cast<float>(best_class);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
      if (out_result_size != static_cast<std::size_t>(preds.size())) {
        throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            "Failed assumption: Treelite was expected to produce an output "
            "that is as long as preds tensor");
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
      for (std::size_t i = 0; i < out_result_size; ++i) {
        preds.data()[i] = out_buffer[i];
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "here");
    }

  } else {
    throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        "Instance kind must be set to either GPU or CPU");
  }
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const int32_t device_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state),
      handle([&]() {
        //if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
          return std::make_unique<raft::handle_t>();
        //} else {
        //  return std::unique_ptr<raft::handle_t>(nullptr);
        //}
      }())
{
  model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId());

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Using GPU for inference");
    ML::fil::from_treelite(
        *handle, &fil_forest, model_state_->treelite_handle,
        &(model_state_->tl_params));
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Using CPU for inference");
  } else {
    throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        "Instance kind must be set to either GPU or CPU");
  }
}

void
ModelInstanceState::UnloadFILModel()
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    ML::fil::free(*handle, fil_forest);
  }
}

}}}  // namespace triton::backend::fil
