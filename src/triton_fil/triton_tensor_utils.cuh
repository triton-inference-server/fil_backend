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
#include <cstddef>
#include <cstdint>
#include <iostream>  // TODO: REMOVE
#include <vector>
#include <raft/handle.hpp>
#include <triton_fil/triton_tensor.cuh>
#include <triton/core/tritonserver.h>
#include <triton_fil/dtype.h>
#include <triton_fil/exceptions.h>
#include <triton_fil/optional.h>

namespace triton { namespace backend { namespace fil {

using byte = char; // C++17: Use std::byte

namespace {
/* Get the number of inputs in each request in a batch
*/
uint32_t get_input_count(
    std::vector<TRITONBACKEND_Request*>& requests,
    bool validate=true) {
  // C++17: Switch to std::optional and remove pragmas
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  optional<uint32_t> input_count;
#pragma GCC diagnostic pop
  for (auto& request : requests) {
    uint32_t cur_input_count = 0;
    triton_check(TRITONBACKEND_RequestInputCount(request, &cur_input_count));
    if (!validate) {
      return cur_input_count;
    }
    if (input_count) {
      // C++17: Switch to std::optional and remove pragmas
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
      if (*input_count != cur_input_count) {
#pragma GCC diagnostic pop
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "inconsistent number of inputs in batched requests"
        );
      }
    } else {
      input_count = cur_input_count;
    }
  }
  if (input_count) {
    return *input_count;
  } else {
    return 0;
  }
}

/* Get the name of a given input
*/
std::string get_input_name(
  uint32_t input_index,
  std::vector<TRITONBACKEND_Request*>& requests,
  bool validate=true
) {

  optional<std::string> input_name;
  for (auto& request : requests) {
    const char* cur_input_name_c = nullptr;
    triton_check(TRITONBACKEND_RequestInputName(
      request, input_index, &cur_input_name_c
    ));
    std::string cur_input_name = cur_input_name_c;
    if (!validate) {
      return cur_input_name;
    }
    if (input_name) {
      if (*input_name != cur_input_name) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "inconsistent input names in batched request"
        );
      }
    } else {
      input_name = cur_input_name;
    }
  }

  if(input_name) {
    return *input_name;
  } else {
    throw TritonException(
      TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
      "no requests given; could not determine input name"
    );
  }
}

/* Return a vector of pointers to TRITONBACKEND_Input objects of the given name
 * for each request
*/
std::vector<TRITONBACKEND_Input*> get_backend_inputs(
  std::string& input_name,
  std::vector<TRITONBACKEND_Request*>& requests
) {
  std::vector<TRITONBACKEND_Input*> all_inputs;
  for (auto& request : requests) {
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
    TRITONSERVER_DataType expected_dtype
) {

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<uint32_t> buffer_pieces;
  for (auto& input : inputs) {

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

    if (input_shapes.size() != 0 && (
      input_shapes[0].size() != cur_dims_count ||
      !std::equal(input_shapes[0].begin() + 1, input_shapes[0].end(), cur_shape + 1))
    ) {
      throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
        "inconsistent input shapes in batched request"
      );
    }
    input_shapes.emplace_back(cur_shape, cur_shape + cur_dims_count);
  }
  return {input_shapes, buffer_pieces};
}

/* Divide input into batches which will minimize device allocations and copies.
 * Each entry in the returned vector represents a complete "batch" and the
 * indices of the requests which are represented in that batch. Batching rules
 * are as follows in order of priority:
 * 
 * 1. If a buffer is contiguous with the previous buffer, combine them into a
 * single buffer in the existing batch. Doing so minimizes the number of copies
 * when it is necessary to copy from host to device or device to device in
 * order to collate a batch.
 * 2. If the first buffer for an input is already on-device, create a new
 * batch. If this new batch ends up with only a single contiguous buffer, it
 * will be processed in-place, avoiding new device allocations and a
 * corresponding copy. This rule ensures that a single input is never split
 * across batches.
 * 3. Otherwise, add the buffer to the existing batch. New device memory for a
 * batch is allocated in a single chunk, so making these batches as large as
 * possible when an allocation is required minimizes the number of allocations.
*/
std::vector<
 std::pair<std::vector<RawInputBuffer>, std::pair<std::size_t, std::size_t>>
> get_raw_input_batches(
  std::vector<TRITONBACKEND_Input*>& all_inputs,
  TRITONSERVER_MemoryType& input_memory_type,
  std::vector<uint32_t>& all_buffer_counts
) {
  std::vector<std::pair<std::vector<RawInputBuffer>, std::vector<std::size_t>>>
    buffer_requests;
  byte* next_ptr = nullptr;
  optional<TRITONSERVER_MemoryType> last_memory_type;

  for (std::size_t r=0; r < all_inputs.size(); ++r) {
    TRITONSERVER_MemoryType memory_type = input_memory_type;

    for (uint32_t j = 0; j < all_buffer_counts[r]; ++j) {
      byte* cur_buffer;
      uint64_t cur_byte_size = 0;
      int64_t cur_memory_type_id = 0;
      triton_check(TRITONBACKEND_InputBuffer(
        all_inputs[r], j,
        const_cast<const void**>(reinterpret_cast<void **>(&cur_buffer)),
        &cur_byte_size, &memory_type, &cur_memory_type_id
      ));
      // Check if this buffer is contiguous with previous
      if (cur_buffer == next_ptr
          && last_memory_type
          && *last_memory_type == memory_type
          && buffer_requests.back().first.device_id == cur_memory_type_id) {
        buffer_requests.back().first.size_bytes += cur_byte_size;
        buffers_requests.back().second.second = r + 1;
      } else {
        // If it's already in device memory, do not batch it with other
        // requests
        if (
            (j == 0 && memory_type == TRITONSERVER_MEMORY_GPU) ||
            buffer_requests.empty()
        ) {
          buffer_requests.push_back({{}, {r, r}});
        }
        buffer_requests.back().first.emplace_back(
          const_cast<const void*>(reinterpret_cast<void*>(cur_buffer)),
          cur_byte_size, memory_type, cur_memory_type_id
        );
        buffer_requests.back().second.second = r + 1;
        last_memory_type = memory_type;
        next_ptr = cur_buffer + cur_byte_size;
      }
    }
  }

  return buffer_requests;
}

/* Given the shape of each input and the request indices covered by a batch,
 * determine the total shape for that batch */
std::vector<int64_t> get_batch_shape(
  std::vector<std::vector<int64_t>>& input_shapes,
  std::pair<std::size_t, std::size_t> batch_extent
) {
  std::vector<int64_t> batch_shape = input_shapes[batch_extent.first];
  for (int i=batch_extent.first + 1; i < batch_extent.second; ++i) {
    batch_shape[0] += input_shapes[i][0];
  }
  return batch_shape;
}

/* Get the name of a given output
*/
std::string get_output_name(
  uint32_t output_index,
  std::vector<TRITONBACKEND_Request*>& requests,
  bool validate=true
) {

  optional<std::string> output_name;
  for (auto& request : requests) {
    const char * cur_output_name_c = nullptr;
    triton_check(TRITONBACKEND_RequestOutputName(
      request, output_index, &cur_output_name_c
    ));
    std::string cur_output_name = cur_output_name_c;
    if (!validate) {
      return cur_output_name;
    }
    if (output_name) {
      if (*output_name != cur_output_name) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          "inconsistent output names in batched request"
        );
      }
    } else {
      output_name = cur_output_name;
    }
  }

  if(output_name) {
    return *output_name;
  } else {
    throw TritonException(
      TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
      "no requests given; could not determine input name"
    );
  }
}

std::vector<TRITONBACKEND_Output*> get_backend_outputs(
  std::string& name,
  std::vector<TRITONBACKEND_Response*>& responses,
  TRITONSERVER_DataType dtype,
  const std::vector<int64_t>& shape
) {
    std::vector<TRITONBACKEND_Output*> all_outputs;
    for (auto& response : responses) {
      TRITONBACKEND_Output* output;
      triton_check(TRITONBACKEND_ResponseOutput(
        response,
        &output,
        name.c_str(),
        dtype,
        shape.data(),
        shape.size()
      ));
      all_outputs.push_back(output);
    }
    return all_outputs;
}

template<TRITONSERVER_DataType D>
std::vector<RawOutputBuffer> get_raw_output_buffers(
  std::vector<TRITONBACKEND_Output*> all_outputs,
  TRITONSERVER_MemoryType& output_memory_type,
  const std::vector<int64_t>& shape,
  int64_t device_id
) {
  std::vector<RawOutputBuffer> buffers;

  byte* next_ptr = nullptr;
  optional<TRITONSERVER_MemoryType> last_memory_type;

  uint64_t byte_size = product(shape) * sizeof(typename TritonType<D>::type) / shape[0];

  for (auto output : all_outputs) {
    byte* cur_buffer;
    int64_t cur_memory_type_id = 0;
    TRITONSERVER_MemoryType memory_type = output_memory_type;
    triton_check(TRITONBACKEND_OutputBuffer(
      output,
      reinterpret_cast<void**>(&cur_buffer),
      byte_size,
      &memory_type,
      &device_id
    ));

    if (cur_buffer == next_ptr
        && last_memory_type
        && *last_memory_type == memory_type
        && buffers.back().device_id == cur_memory_type_id) {
      buffers.back().size_bytes += byte_size;
    } else {
      buffers.emplace_back(
        reinterpret_cast<void*>(cur_buffer), static_cast<uint64_t>(byte_size),
        memory_type, cur_memory_type_id
      );
      last_memory_type = memory_type;
      next_ptr = cur_buffer + byte_size;
    }
  }
  return buffers;
}

} // anonymous namespace

template<typename T>
std::vector<
  std::pair<TritonTensor<const T>, std::pair<std::size_t, std::size_t>>
> get_input_batches(
  uint32_t input_index,
  std::vector<TRITONBACKEND_Request*>& requests,
  TRITONSERVER_memorytype_enum input_memory_type,
  raft::handle_t& handle,
  bool validate=false
) {
  cudaStream_t cuda_stream = handle.get_stream();
  // Name of input
  auto input_name = get_input_name(input_index, requests, validate);
  // Objects representing input for each request that can be queried to get
  // underlying data and properties
  auto backend_inputs = get_backend_inputs(input_name, requests);

  // Shape of each input tensor
  std::vector<std::vector<int64_t>> input_shapes;
  // How many buffers the data are broken up into for each request
  std::vector<uint32_t> buffer_counts;
  std::tie(total_shape, buffer_counts) = get_input_shapes(
    backend_inputs, TritonDtype<T>::value
  );

  // Pointers to underlying contiguous buffers along with their size and what
  // device they are stored on, divided up into optimal batches. The first and
  // last indices of the corresponding requests are indicated by the second
  // element in each pair representing the batch.
  auto raw_batches = get_raw_input_batches(
    backend_inputs, input_memory_type, buffer_counts
  );

  std::vector<
    std::pair<TritonTensor<const T>, std::pair<std::size_t, std::size_t>>
  > batches;

  for (auto& raw_batch : raw_batches) {
    auto batch_shape = get_batch_shape(input_shapes, raw_batch.second);
    batches.push_back(
      TritonTensor<const T>(
        raw_batch.first, input_name, batch_shape, TritonDtype<T>::value, cuda_stream
      ),
      raw_batch.second
    );
  }

  std::cout << "TOTAL BATCHES: " << batches.size() << "\n";

  return batches;
}

template <typename T>
TritonTensor<T> get_output_batch(
  uint32_t output_index,
  std::vector<TRITONBACKEND_Request*>& requests,
  std::vector<TRITONBACKEND_Response*>& responses,
  TRITONSERVER_MemoryType output_memory_type,
  const std::vector<int64_t>& tensor_shape,
  raft::handle_t& handle
  // bool validate=false TODO
) {
  bool validate=false;
  cudaStream_t cuda_stream = handle.get_stream();
  // Name of output
  auto output_name = get_output_name(output_index, requests, validate);
  // Objects representing output for each request that can be queried to get
  // underlying data and properties
  auto backend_outputs = get_backend_outputs(
    output_name, responses, TritonDtype<T>::value, tensor_shape);

  // Pointers to underlying contiguous buffers along with their size and what
  // device they are stored on
  auto raw_buffers = get_raw_output_buffers<TritonDtype<T>::value>(
    backend_outputs, output_memory_type, tensor_shape, static_cast<int64_t>(0)
  );

  return TritonTensor<T>(
    std::move(raw_buffers), output_name, tensor_shape, TritonDtype<T>::value, handle
  );
}

}}}
