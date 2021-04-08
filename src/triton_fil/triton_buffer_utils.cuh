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
