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
#include <algorithm>
#include <cstring>
#include <optional>

#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <triton_fil/model_state.h>
#include <triton_fil/gtil_utils.cuh>
#include <triton_fil/model_instance_state.cuh>
#include <triton_fil/triton_tensor.cuh>

namespace triton { namespace backend { namespace fil {
void
gtil_predict(
    ModelInstanceState& instance_state, TritonTensor<const float>& data,
    TritonTensor<float>& preds, bool predict_proba)
{
  auto& model_state = [&]() -> ModelState& {
    auto model_ptr = instance_state.StateForModel();
    if (model_ptr == nullptr) {
      throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "No model associated with instance");
    }
    return *model_ptr;
  }();

  std::size_t num_class = 1;
  auto tl_model = static_cast<treelite::Model*>(model_state.treelite_handle);
  num_class = tl_model->task_param.num_class;

  float* out_preds = preds.data();
  // Use a temporary buffer when outputing class outputs from a
  // multi-class classifier. In this case, preds will have length
  // num_row, whereas GTIL expects a buffer of length
  // (num_row * num_class). Thus, a temporary buffer is needed to
  // avoid out-of-bounds access.
  std::optional<std::vector<float>> temp_buffer = std::nullopt;
  if (!predict_proba && model_state.tl_params.output_class && num_class > 1) {
    std::strcpy(tl_model->param.pred_transform, "max_index");
    temp_buffer = std::vector<float>(data.shape()[0] * num_class);
    out_preds = temp_buffer->data();
  }
  std::size_t out_result_size;
  int gtil_result = TreeliteGTILPredict(
      model_state.treelite_handle, data.data(), data.shape()[0], out_preds, 1,
      &out_result_size);
  if (gtil_result != 0) {
    throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
        TreeliteGetLastError());
  }
  if (temp_buffer) {
    std::copy(out_preds, out_preds + preds.size(), preds.data());
  }

  if (num_class == 1 && predict_proba) {
    std::size_t i = preds.size();
    while (i > 0) {
      --i;
      preds.data()[i] =
          ((i % 2) == 1) ? preds.data()[i / 2] : 1.0f - preds.data()[i / 2];
    }
  } else if (
      num_class == 1 && !predict_proba && model_state.tl_params.output_class) {
    std::transform(
        preds.data(), preds.data() + preds.size(), preds.data(),
        [&](float raw_pred) {
          return (raw_pred > model_state.tl_params.threshold) ? 1.0f : 0.0f;
        });
  }
}
}}}  // namespace triton::backend::fil
