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
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_buffer.cuh>

namespace triton { namespace backend { namespace fil {

std::unique_ptr<ModelInstanceState>
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
{
  std::string instance_name = get_model_instance_name(*triton_model_instance);
  TRITONSERVER_InstanceGroupKind instance_kind =
      get_instance_kind(*triton_model_instance);
  int32_t instance_id = get_device_id(*triton_model_instance);

  return std::make_unique<ModelInstanceState>(
      model_state, triton_model_instance, instance_name.c_str(), instance_kind,
      instance_id);
}

raft::handle_t&
ModelInstanceState::get_raft_handle()
{
  return *handle;
}

void
ModelInstanceState::predict(
    TritonTensor<const float>& data, TritonTensor<float>& preds, size_t num_rows,
    bool predict_proba)
{
  try {
    ML::fil::predict(
        *handle, fil_forest, preds.data(), data.data(), num_rows,
        predict_proba);
  }
  catch (raft::cuda_error& err) {
    throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL, err.what());
  }
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), handle(std::make_unique<raft::handle_t>())
{
  model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId());
  ML::fil::from_treelite(
      *handle, &fil_forest, model_state_->treelite_handle,
      &(model_state_->tl_params));
}

void
ModelInstanceState::UnloadFILModel()
{
  ML::fil::free(*handle, fil_forest);
}

}}}  // namespace triton::backend::fil
