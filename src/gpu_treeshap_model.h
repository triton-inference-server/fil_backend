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

#include <cuda_runtime_api.h>
#include <tl_model.h>
#include <treeshap_model.h>

#include <cstddef>
#include <cuml/explainer/tree_shap.hpp>
#include <memory>
#include <raft/handle.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

template <>
struct TreeShapModel<rapids::DeviceMemory> {
  using device_id_t = int;
  TreeShapModel(
      device_id_t device_id, cudaStream_t stream,
      std::shared_ptr<TreeliteModel> tl_model)
      : device_id_{device_id}, raft_handle_{stream}, tl_model_{tl_model},
        path_info_{ML::Explainer::extract_path_info(tl_model_->handle())}
  {
  }

  TreeShapModel(TreeShapModel const& other) = default;
  TreeShapModel& operator=(TreeShapModel const& other) = default;
  TreeShapModel(TreeShapModel&& other) = default;
  TreeShapModel& operator=(TreeShapModel&& other) = default;

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t n_rows, std::size_t n_cols) const
  {
    // Need to synchronize on the stream because treeshap currently does not
    // take a stream on its API
    input.stream_synchronize();
    ML::Explainer::gpu_treeshap(
      path_info_,
      ML::Explainer::FloatPointer(const_cast<float*>(input.data())),
      n_rows,
      n_cols,
      ML::Explainer::FloatPointer(output.data()),
      output.size()
    );
    output.stream_synchronize();
  }

 private:
  raft::handle_t raft_handle_;
  std::shared_ptr<TreeliteModel> tl_model_;
  device_id_t device_id_;
  ML::Explainer::TreePathHandle path_info_;
};

}}}  // namespace triton::backend::NAMESPACE
