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
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <rapids::MemoryType M>
struct TreeShapModel {
  using device_id_t = int;

  TreeShapModel(std::shared_ptr<TreeliteModel> tl_model)
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "TreeShapModel invoked with a memory type unsupported by this build");
  }
  TreeShapModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model)

  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "TreeShapModel invoked with a memory type unsupported by this build");
  }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t n_rows, std::size_t n_cols) const
  {
    throw rapids::TritonException(
        rapids::Error::Unsupported,
        "TreeShapModel invoked with a memory type unsupported by this build");
  }
};
}}}  // namespace triton::backend::NAMESPACE
