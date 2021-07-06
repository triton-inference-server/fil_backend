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
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_tensor.cuh>

namespace triton { namespace backend { namespace fil {
void gtil_predict(
    ModelInstanceState& instance_state, TritonTensor<const float>& data,
    TritonTensor<float>& preds, bool predict_proba);
}}}  // namespace triton::backend::fil
