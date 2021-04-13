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
#include <vector>
#include <raft/handle.hpp>
#include <triton_fil/triton_buffer.cuh>

namespace triton { namespace backend { namespace fil {

template<typename T>
std::vector<TritonBuffer<const T>> get_input_buffers(
    TRITONBACKEND_Request* request,
    TRITONSERVER_MemoryType input_memory_type,
    raft::handle_t& raft_handle) {

  uint32_t input_count = 0;
  triton_check(TRITONBACKEND_RequestInputCount(
    request, &input_count));

  std::vector<TritonBuffer<const T>> buffers;
  // buffers.reserve(input_count); TODO

  for (uint32_t i = 0; i < input_count; ++i) {
    const char* input_name;
    triton_check(TRITONBACKEND_RequestInputName(
      request, i, &input_name));

    TRITONBACKEND_Input* input = nullptr;
    triton_check(TRITONBACKEND_RequestInput(
        request, input_name, &input));

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    triton_check(TRITONBACKEND_InputProperties(
        input, nullptr, &input_datatype, &input_shape, &input_dims_count,
        &input_byte_size, &input_buffer_count));

    for (uint32_t j = 0; j < input_buffer_count; ++j) {
      const void * input_buffer = nullptr;
      uint64_t buffer_byte_size = 0;
      int64_t input_memory_type_id = 0;

      triton_check(TRITONBACKEND_InputBuffer(
          input, j, &input_buffer, &buffer_byte_size, &input_memory_type,
          &input_memory_type_id));

      std::vector<int64_t> shape(input_shape, input_shape + input_dims_count);

      if (input_memory_type == TRITONSERVER_MEMORY_GPU) {
        buffers.emplace_back(
          reinterpret_cast<const T*>(input_buffer),
          std::string(input_name),
          shape,
          input_datatype
        );
      } else {
        buffers.emplace_back(
          reinterpret_cast<const T*>(input_buffer),
          std::string(input_name),
          shape,
          input_datatype,
          raft_handle.get_stream()
        );
      }
    }
  }
  return buffers;
}

template<typename T>
std::vector<TritonBuffer<T>> get_output_buffers(
    TRITONBACKEND_Request* request,
    TRITONBACKEND_Response* response,
    TRITONSERVER_MemoryType memory_type,
    TRITONSERVER_DataType dtype,
    const std::vector<int64_t>& shape,
    raft::handle_t& raft_handle) {
  uint32_t count = 0;
  triton_check(TRITONBACKEND_RequestOutputCount(
      request, &count));

  std::vector<TritonBuffer<T>> buffers;
  buffers.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    const char* name;
    triton_check(TRITONBACKEND_RequestOutputName(request, i, &name));

    TRITONBACKEND_Output* output;
    triton_check(TRITONBACKEND_ResponseOutput(
        response,
        &output,
        name,
        dtype,
        shape.data(),
        shape.size()));

    int64_t memory_type_id = 0;

    void * output_buffer;
    auto element_count = std::accumulate(std::begin(shape),
                                         std::end(shape),
                                         1,
                                         std::multiplies<>());
    uint64_t buffer_byte_size = element_count * sizeof(T);
    triton_check(TRITONBACKEND_OutputBuffer(
        output,
        &output_buffer,
        buffer_byte_size,
        &memory_type,
        &memory_type_id));

    if (memory_type == TRITONSERVER_MEMORY_GPU) {
      buffers.emplace_back(
        reinterpret_cast<T*>(output_buffer),
        std::string(name),
        shape,
        dtype
      );
    } else {
      buffers.emplace_back(
        std::string(name),
        shape,
        dtype,
        raft_handle.get_stream(),
        reinterpret_cast<T*>(output_buffer)
      );
    }
  }
  return buffers;
}

}}}
