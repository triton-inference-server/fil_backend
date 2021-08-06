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
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>

#include <numeric>
#include <rmm/mr/device/per_device_resource.hpp>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace fil {

/**
 * @brief Return the product of all elements in a vector
 */
template <typename T>
T
product(const std::vector<T>& array)
{
  return std::accumulate(
      std::begin(array), std::end(array), 1, std::multiplies<>());
}

/**
 * @brief Allocate given number of elements on GPU and return device pointer
 */
template <typename T>
T*
allocate_device_memory(size_t count, cudaStream_t stream)
{
  auto* ptr_d =
      static_cast<T*>(rmm::mr::get_current_device_resource()->allocate(
          sizeof(T) * count, stream));
  return ptr_d;
}

/**
 * @brief Struct for storing pointer to Triton input data and associated
 * metadata
 * RawInputBuffers are intended to store all metadata provided by Triton about
 * a single input to a backend in a unified struct.
 */
struct RawInputBuffer {
  /** Pointer to where data is stored */
  const void* data;
  /** Number of bytes of data stored in this buffer */
  uint64_t size_bytes;
  /** Enum indicating whether data is stored on GPU or host */
  TRITONSERVER_MemoryType memory_type;
  /** ID of GPU on which data is stored; 0 if data is stored on host */
  int64_t device_id;
  RawInputBuffer(
      const void* data, uint64_t size_bytes,
      TRITONSERVER_MemoryType& memory_type, int64_t device_id)
      : data(data), size_bytes(size_bytes), memory_type(memory_type),
        device_id(device_id)
  {
  }
};

/**
 * @brief Struct for storing pointer to where Triton expects final output data
 * to be stored along with associated metadata
 * RawInputBuffers are intended to store all metadata provided by Triton about
 * a single output from a backend in a unified struct.
 */
struct RawOutputBuffer {
  /** Pointer to where output data should be stored */
  void* data;
  /** Capacity of this output buffer in bytes */
  uint64_t size_bytes;
  /** Enum indicating whether data is stored on GPU or host */
  TRITONSERVER_MemoryType memory_type;
  /** ID of GPU on which data is stored; 0 if data is stored on host */
  int64_t device_id;

  RawOutputBuffer(
      void* data, uint64_t size_bytes, TRITONSERVER_MemoryType& memory_type,
      int64_t device_id)
      : data(data), size_bytes(size_bytes), memory_type(memory_type),
        device_id(device_id)
  {
  }
};

}}}  // namespace triton::backend::fil
