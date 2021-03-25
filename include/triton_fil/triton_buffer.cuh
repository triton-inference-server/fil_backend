#pragma once
#include <numeric>
#include <string>
#include <vector>
#include <raft/handle.hpp>
#include <raft/cudart_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.hpp>

namespace triton { namespace backend { namespace fil {

template<typename T>
struct TritonBuffer {
  std::string name;
  std::vector<int64_t> shape;
  TRITONSERVER_DataType dtype;
  uint64_t byte_size;
  T* buffer;
  TRITONSERVER_MemoryType memory_type;
  bool requires_deallocation;

  T* get_data() {
    return buffer;
  }

  ~TritonBuffer() {
    /* if (requires_deallocation) {
      CUDA_CHECK(cudaFree(buffer));
    } */
  }
};

template<typename T>
std::vector<TritonBuffer<T>> get_input_buffers(
    TRITONBACKEND_Request* request,
    TRITONSERVER_MemoryType input_memory_type,
    raft::handle_t& raft_handle) {
  uint32_t input_count = 0;
  TRITONBACKEND_RequestInputCount(
      request, &input_count);  // TODO: error handling

  std::vector<TritonBuffer<T>> buffers;
  buffers.reserve(input_count);

  for (uint32_t i = 0; i < input_count; ++i) {
    const char* input_name;
    TRITONBACKEND_RequestInputName(
        request, i, &input_name);  // TODO: error handling

    TRITONBACKEND_Input* input = nullptr;
    TRITONBACKEND_RequestInput(
        request, input_name, &input);  // TODO: error handling

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    TRITONBACKEND_InputProperties(  // TODO: error handling
        input, nullptr, &input_datatype, &input_shape, &input_dims_count,
        &input_byte_size, &input_buffer_count);

    const void * input_buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    int64_t input_memory_type_id = 0;

    TRITONBACKEND_InputBuffer(
        input, i, &input_buffer, &buffer_byte_size, &input_memory_type,
        &input_memory_type_id);  // TODO: error handling

    std::vector<int64_t> shape(input_shape, input_shape + input_dims_count);
    T* final_buffer;
    bool requires_deallocation;
    if (input_memory_type == TRITONSERVER_MEMORY_GPU) {
      final_buffer = (T*) input_buffer;
      requires_deallocation = false;
    } else {
      raft::allocate(final_buffer, buffer_byte_size);
      raft::copy(final_buffer, (T*) input_buffer, buffer_byte_size / sizeof(T),
          raft_handle.get_stream());
      requires_deallocation = true;
    }
    buffers.push_back(TritonBuffer<T>{input_name,
                                      shape,
                                      input_datatype,
                                      buffer_byte_size,
                                      final_buffer,
                                      TRITONSERVER_MEMORY_GPU,
                                      requires_deallocation});
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
  TRITONBACKEND_RequestOutputCount(
      request, &count);  // TODO: error handling

  std::vector<TritonBuffer<T>> buffers;
  buffers.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    const char* name;
    TRITONBACKEND_RequestOutputName(request, i, &name);  // TODO: error handling

    TRITONBACKEND_Output* output;
    TRITONBACKEND_ResponseOutput(
        response,
        &output,
        name,
        dtype,
        shape.data(),
        shape.size()); // TODO: error handling

    int64_t memory_type_id = 0;

    void * output_buffer;
    auto element_count = std::accumulate(std::begin(shape),
                                         std::end(shape),
                                         1,
                                         std::multiplies<>());
    uint64_t buffer_byte_size = element_count * sizeof(T);
    TRITONBACKEND_OutputBuffer(
        output,
        &output_buffer,
        buffer_byte_size,
        &memory_type,
        &memory_type_id);  // TODO: error handling

    T* final_buffer;
    final_buffer = (T*) output_buffer;
    bool requires_deallocation = false;

    buffers.push_back(TritonBuffer<T>{name,
                                      shape,
                                      dtype,
                                      buffer_byte_size,
                                      final_buffer,
                                      memory_type,
                                      requires_deallocation});
  }
  return buffers;
}

}}}