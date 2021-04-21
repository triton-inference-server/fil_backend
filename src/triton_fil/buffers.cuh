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
#include <numeric>
#include <string>
#include <vector>
#include <raft/cudart_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>

namespace triton { namespace backend { namespace fil {

template<typename T>
T product(const std::vector<T>& array) {
  return std::accumulate(std::begin(array),
                         std::end(array),
                         1,
                         std::multiplies<>());
}

template<typename T>
T* allocate_device_memory(size_t bytes) {
  T* ptr_d;
  raft::allocate(ptr_d, bytes);
  return ptr_d;
}

struct RawInputBuffer {
  const void* data;
  uint64_t size_bytes;
  TRITONSERVER_MemoryType memory_type;
  int64_t device_id;
  RawInputBuffer(
    const void* data, uint64_t size_bytes,
    TRITONSERVER_MemoryType& memory_type, int64_t device_id
  ) : data(data), size_bytes(size_bytes), memory_type(memory_type),
  device_id(device_id) {}
};

struct RawOutputBuffer {
  void* data;
  uint64_t size_bytes;
  TRITONSERVER_MemoryType memory_type;
  int64_t device_id;

  RawOutputBuffer(
    void* data, uint64_t size_bytes,
    TRITONSERVER_MemoryType& memory_type, int64_t device_id
  ) : data(data), size_bytes(size_bytes), memory_type(memory_type),
  device_id(device_id) {}
};

}}}
