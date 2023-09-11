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

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#include <names.h>
#include <tl_model.h>

#include <cstddef>
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

/* This struct defines a unified prediction interface to both FIL and GTIL.
 * Template specializations are provided based on the type of memory the model
 * is expected to process */
template <rapids::MemoryType M>
struct ForestModel {
  using device_id_t = int;

  ForestModel(std::shared_ptr<TreeliteModel> tl_model, bool use_new_fil)
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "ForestModel invoked with a memory type unsupported by this build");
  }

  ForestModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model, bool use_new_fil)
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "ForestModel invoked with a memory type unsupported by this build");
  }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t samples, bool predict_proba) const
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "ForestModel invoked with a memory type unsupported by this build");
  }
};

// TODO(hcho3): Remove this once raft_proto becomes part of RAFT or
// Rapids-Triton
raft_proto::device_type
get_raft_proto_device_type(rapids::MemoryType mem_type)
{
  if (mem_type == rapids::DeviceMemory) {
    return raft_proto::device_type::gpu;
  } else {
    return raft_proto::device_type::cpu;
  }
}

}}}  // namespace triton::backend::NAMESPACE
