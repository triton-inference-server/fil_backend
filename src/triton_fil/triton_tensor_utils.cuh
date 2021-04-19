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
/* Get the number of inputs in each request in a batch
*/
uint32_t get_input_count(
    std::vector<TRITONBACKEND_Request*>& requests,
    bool validate=true) {
  optional<uint32_t> input_count();
  for (auto& request : requests) {
    uint32_t cur_input_count = 0;
    triton_check(TRITONBACKEND_RequestInputCount(request, &cur_input_count));
    if (!validate) {
      return cur_input_count;
    }
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

/* Get the name of a given index
*/
std::string get_input_name(
  uint32_t input_index
  std::vector<TRITONBACKEND_Request*>& requests,
  bool validate=true;
) {

  optional<std::string> input_name();
  for (auto& request : requests) {
    std::string cur_input_name;
    triton_check(TRITONBACKEND_RequestInputName(
      request, input_index, cur_input_name.c_str()
    ));
    if (!validate) {
      return cur_input_name;
    }
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

  if(input_name) {
    return *input_name
  } else {
    throw TritonException(
      TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
      "no requests given; could not determine input name"
    )
  }
}

/* Return a vector of pointers to TRITONBACKEND_Input objects of the given name
 * for each request
*/
std::vector<TRITONBACKEND_Input* input> get_backend_inputs(
  std::string& input_name
  std::vector<TRITONBACKEND_Request*>& requests,
) {
  std::vector<TRITONBACKEND_Input*> all_inputs;
  for (auto request& : requests) {
    TRITONBACKEND_Input* input = nullptr;
    triton_check(TRITONBACKEND_RequestInput(request, input_name.c_str(), &input));
    all_inputs.push_back(input);
  }
  return all_inputs;
}


/* Return a pair whose first element is a vector representing the tensor shape
 * for the entire batch and whose second element is a vector representing how
 * many buffers that tensor is broken into
*/
std::pair<std::vector<std::vector<int64_t>>, std::vector<uint32_t>> get_input_shapes(
    std::vector<TRITONBACKEND_Input*>& inputs,
    uint32_t input_count,
    TRITONSERVER_Datatype& expected_dtype
) {

  std::vector<int64_t> input_shape;
  std::vector<uint32_t> buffer_pieces;
  for (auto& input : all_inputs) {

    TRITONSERVER_DataType cur_dtype;
    const int64_t* cur_shape;
    uint32_t cur_dims_count;
    uint32_t cur_buffer_count;
    triton_check(TRITONBACKEND_InputProperties(
      input, nullptr, &cur_dtype, &cur_shape, &cur_dims_count, nullptr,
      &cur_buffer_count
    ));

    if (cur_dtype != expected_dtype) {
      throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
        "unexpected input data type in batched request"
      );
    }

    buffer_pieces.push_back(cur_buffer_count);

    if (input_shape.size() != 0) {
      if (input_shape->size() != cur_dims_count ||
          !std::equal(input_shape.begin() + 1, input_shape.end(), cur_shape + 1)) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "inconsistent input shapes in batched request"
        );
      }
      input_shape[0] += *cur_shape;
    } else {
      input_shape.assign(cur_shape, cur_shape + cur_dims_count);
    }
  }
  return {input_shape, buffer_pieces};
}

struct RawTritonTensor {
  const void* data;
  uint64_t size_bytes;
  TRITONBACKEND_MemoryType memory_type;
  int64_t device_id;
};

/* For each request, return a vector of all raw buffers that will be combined
 * to form the corresponding input tensor
*/
std::vector<RawTritonTensor> get_raw_input_buffers(
  std::vector<TRITONBACKEND_Input*>& all_inputs,
  TRITONBACKEND_MemoryType& input_memory_type,
  std::vector<uint32_t>& all_buffer_counts
) {
  std::vector<RawTritonTensor> input_buffers;
  void * next_ptr = nullptr;
  optional<TRITONBACKEND_MemoryType> last_memory_type();
  for (size_t r=0; r < all_inputs.size(); ++r) {
    TRITONBACKEND_MemoryType memory_type = input_memory_type;

    for (uint32_t j = 0; j < all_buffer_counts[r]; ++j) {
      const void * cur_buffer;
      uint64_t cur_byte_size = 0;
      int64_t cur_memory_type_id = 0;
      triton_check(TRITONBACKEND_InputBuffer(
        all_inputs[r], j, &cur_buffer, &cur_byte_size, &memory_type,
        &cur_memory_type_id
      ));
      // Check if this buffer is contiguous with previous
      if (cur_buffer == next_ptr
          && last_memory_type
          && *last_memory_type == memory_type
          && input_buffers.back().device_id == cur_memory_type_id) {
        input_buffers.back().size_bytes += cur_size_bytes;
      } else {
        cur_input_buffers.emplace_back(
          cur_buffer, cur_size_bytes, cur_memory_type_id, memory_type
        );
        last_memory_type = optional<TRITONBACKEND_MemoryType>(memory_type);
        next_ptr = cur_buffer + cur_size_bytes;
      }
    }
    input_buffers.push_back(std::move(cur_input_buffers));
  }

  return input_buffers;
}

template<typename T>
TritonTensor<const T> build_input_buffer(
  std::vector<RawTritonTensor>& raw_buffers,
  std::vector<int64_t>& tensor_shape
) {
}

} // anonymous namespace

template<typename T>
TritonTensor<const T> get_input_batch(
  uint32_t input_index,
  std::vector<TRITONBACKEND_Request*>& requests,
  TRITONBACKEND_MemoryType input_memory_type,
  raft::handle_t& raft_handle,
  bool validate=false
) {
  // Name of input
  auto input_name = get_input_name(input_index, requests, validate);
  // Objects representing input for each request that can be queried to get
  // underlying data and properties
  auto backend_inputs = get_backend_inputs(input_name, requests, validate);

  // Shape of each input tensor
  std::vector<int64_t> tensor_shape;
  // How many buffers the input tensor is broken up into for each request
  std::vector<uint32_t> buffer_counts;
  // TODO: pass expected dtype
  std::tie(tensor_shape, buffer_counts) = get_input_shapes(backend_inputs, input_count);

  // Pointers to underlying contiguous buffers along with their size and what
  // device they are stored on
  auto raw_buffers = get_raw_input_buffers(
    backend_inputs, input_memory_type, buffer_counts
  );

  return build_input_buffer(raw_buffers, tensor_shapes);  // TODO
}

template<typename T>
std::vector<TritonTensor<const T>> get_input_buffers(
    TRITONBACKEND_Request* request,
    TRITONSERVER_MemoryType input_memory_type,
    raft::handle_t& raft_handle) {

  uint32_t input_count = 0;
  triton_check(TRITONBACKEND_RequestInputCount(
    request, &input_count));

  std::vector<TritonTensor<const T>> buffers;
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
std::vector<TritonTensor<T>> get_output_buffers(
    TRITONBACKEND_Request* request,
    TRITONBACKEND_Response* response,
    TRITONSERVER_MemoryType memory_type,
    TRITONSERVER_DataType dtype,
    const std::vector<int64_t>& shape,
    raft::handle_t& raft_handle) {
  uint32_t count = 0;
  triton_check(TRITONBACKEND_RequestOutputCount(
      request, &count));

  std::vector<TritonTensor<T>> buffers;
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
