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

#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>
#include <triton_fil/model_instance_state.h>
#include <triton_fil/model_state.h>

#include <memory>
#include <raft/handle.hpp>

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

TRITONSERVER_Error*
ModelInstanceState::predict(const float* data, float* preds, size_t num_rows)
{
  ML::fil::predict(
      *handle, fil_forest, preds, data, num_rows, model_state_->predict_proba);
  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), handle(std::make_unique<raft::handle_t>())
{
  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId()));
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
