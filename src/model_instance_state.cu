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
#include <triton_fil/gtil_utils.cuh>

#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <memory>
#include <optional>
#include <raft/handle.hpp>
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

std::optional<raft::handle_t>&
ModelInstanceState::get_raft_handle()
{
  return handle;
}

void
ModelInstanceState::predict(
    TritonTensor<const float>& data, TritonTensor<float>& preds,
    bool predict_proba)
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    try {
      ML::fil::predict(
          handle.value(), fil_forest, preds.data(), data.data(),
          data.shape()[0], predict_proba);
    }
    catch (raft::cuda_error& err) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL, err.what());
    }
    catch (std::bad_optional_access& err) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "No RAFT handle created in GPU instance");
    }
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    gtil_predict(*this, data, preds, predict_proba);
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
      model_state_(model_state), handle([&]() -> std::optional<raft::handle_t> {
        if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
          return std::make_optional<raft::handle_t>();
        } else {
          return std::nullopt;
        }
      }())
{
  model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId());

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Using GPU for inference");
    ML::fil::from_treelite(
        handle.value(), &fil_forest, model_state_->treelite_handle,
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
