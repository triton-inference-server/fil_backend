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
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>

namespace triton { namespace backend { namespace fil {

namespace {
/** Get the number of inputs in each request in a batch
*/
uint32_t get_input_count(std::vector<TRITONBACKEND_Request*>& requests) {
  optional<uint32_t> input_count();
  for (auto& request : requests) {
    uint32_t cur_input_count = 0;
    triton_check(TRITONBACKEND_RequestInputCount(request, &cur_input_count));
    if (input_count) {
      if (*input_count != input_count) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "inconsistent number of inputs in batched requests"
        )
      }
    } else {
      input_count = optional<uint32_t>(cur_input_count);
    }
  }
  if (input_count) {
    return *input_count;
  } else {
    return 0;
  }
}

/** Get the names of each input in a batch of requests
*/
std::vector<std::string> get_input_names(
    std::vector<TRITONBACKEND_Request*>& requests,
    uint32_t input_count
) {
  std::vector<std::string> input_names();
  input_names.reserve(input_count);

  for (uint32_t i = 0; i < input_count; ++i) {
    optional<std::string> input_name();
    for (auto& request : requests) {
      std::string cur_input_name;
      triton_check(TRITONBACKEND_RequestInputName(
        request, i, cur_input_name.c_str()
      ));
      if (input_name) {
        if (*input_name != cur_input_name) {
          throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            "inconsistent input names in batched request"
          )
        }
      } else {
        input_name = optional<uint32_t>(cur_input_name);
      }
    }
    input_names.push_back(input_name.value_or(""));
  }
  return input_names;
}

/** Return a vector containing one vector for each input. Each element of the
 * inner vectors contains a pointer to the corresponding TRITONBACKEND_Input
 * object for each request
*/
std::vector<std::vector<TRITONBACKEND_Input* input>> get_backend_inputs(
  std::vector<TRITONBACKEND_Request*>& requests,
  std::vector<std::string>& input_names
) {
  std::vector<TRITONBACKEND_Input*> all_inputs;
  for (auto name& : input_names) {
    for (auto request& : requests) {
      TRITONBACKEND_Input* input = nullptr;
      triton_check(TRITONBACKEND_RequestInput(request, name.c_str(), &input));
      all_inputs.push_back(input);
    }
  }
  return all_inputs;
}


/** Return a vector containing a pair for each input. The first element in each
 * pair is a vector representing the shape of the tensor for that input. The
 * second element is a vector containing the number of buffers that the tensor
 * is broken up into for each request.
*/
std::vector<std::pair<std::vector<int64_t>, std::vector<uint32_t>>> get_input_shapes(
    std::vector<TRITONBACKEND_Input*>& inputs,
    uint32_t input_count,
    const std::vector<TRITONSERVER_Datatype>& expected_dtypes
) {
  std::vector<
    std::pair<std::vector<int64_t>, std::vector<uint32_t>>
  > input_shapes;
  for (uint32_t i = 0; i < input_count; ++i) {
    std::vector<int64_t> input_shape;
    std::vector<uint32_t> buffer_pieces;
    for (auto& input : all_inputs[i]) {

      TRITONSERVER_DataType cur_dtype;
      const int64_t* cur_shape;
      uint32_t cur_dims_count;
      uint32_t cur_buffer_count;
      triton_check(TRITONBACKEND_InputProperties(
        input, nullptr, &cur_dtype, &cur_shape, &cur_dims_count, nullptr,
        &cur_buffer_count
      ));

      if (cur_dtype != expected_dtypes[i]) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "unexpected input data type in batched request"
        );
      }

      buffer_pieces.push_back(cur_buffer_count);
      if (input_shape.size() != 0) {
        if (input_shape->size() != cur_dims_count ||
            !std::equal(input_shape.begin(), input_shape.end(), cur_shape)) {
          throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            "inconsistent input shapes in batched request"
          );
        }
      } else {
        input_shape.insert(
            input_shape.end(), cur_shape, cur_shape + cur_dims_count
        );
      }
    }

    input_shapes.emplace_back(std::move(input_shape), std::move(buffer_pieces));
  }

  return input_shapes;
}

/** For each input, return a vector of pairs whose first element is a pointer
 * to the underlying data and whose second pair is the size of that chunk of
 * data in bytes
*/
template<typename T>
std::vector<std::vector<std::pair<const void*, uint64_t>>> get_raw_input_buffers(
  std::vector<TRITONBACKEND_Input*>& all_inputs,
  uint32_t input_count,
  std::vector<TRITONBACKEND_MemoryType> input_memory_type,
  std::vector<std::pair<std::vector<int64_t>,
                        std::vector<uint32_t>>> input_shapes
) {
  for (uint32_t i = 0; i < input_count; ++i) {
    for (auto& input : all_inputs[i]) {
      optional<TRITONBACKEND_MemoryType> cur_memory_type();
      for (uint32_t j = 0; j < input_buffer_count; ++j) {
        const void * cur_buffer;
        uint64_t cur_byte_size = 0;
        int64_t cur_memory_type_id = 0;
        triton_check(TRITONBACKEND_InputBuffer(
          input, j, &cur_buffer, &cur_byte_size, &input_memory_type[i],
          &cur_memory_type_id
        ));
      }
    }
  }
}


} // anonymous namespace

TritonBuffer<const T> get_input_batch(
  std::vector<TRITONBACKEND_Request*>& requests,
  std::vector<TRITONBACKEND_MemoryType> input_memory_type,
  raft::handle_t& raft_handle
) {
  auto input_count = get_input_count(requests);
  auto input_names = get_input_names(requests, input_count);
  auto backend_inputs = get_backend_inputs(requests, input_count, input_names);
  // TODO: pass expected dtype
  auto input_shapes = get_input_shapes(backend_inputs, input_count);
}

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
