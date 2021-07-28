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
#include <cuml/fil/fil.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/model_state.h>

#include <iterator>
#include <memory>
#include <optional>
#include <raft/handle.hpp>
#include <string>

namespace triton { namespace backend { namespace fil {

template <typename T>
class TritonTensor;

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static std::unique_ptr<ModelInstanceState> Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  void UnloadFILModel();
  void predict(
      TritonTensor<const float>& data, TritonTensor<float>& preds,
      bool predict_proba = false);

  std::optional<raft::handle_t>& get_raft_handle();

  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const int32_t device_id);

  template <class Iter>
  typename std::enable_if_t<std::is_same<
      typename std::iterator_traits<Iter>::value_type,
      TRITONBACKEND_Request*>::value>
  report_statistics(
      Iter requests_begin, Iter requests_end, bool success,
      uint64_t start_time, uint64_t compute_start_time, uint64_t compute_end_time,
      uint64_t end_time){
    try {
      triton::backend::fil::report_statistics(
        *triton_model_instance_, requests_begin, requests_end, success,
        start_time, compute_start_time, compute_end_time, end_time
      );
    }
    catch (TritonException& stat_err) {
      log_error(__FILE__, __LINE__, stat_err.what());
    }
  }

  void report_statistics(
      std::size_t inference_count, uint64_t start_time, uint64_t
      compute_start_time, uint64_t compute_end_time, uint64_t end_time);

 private:
  ModelState* model_state_;
  ML::fil::forest_t fil_forest;
  std::optional<raft::handle_t> handle;
};

}}}  // namespace triton::backend::fil
