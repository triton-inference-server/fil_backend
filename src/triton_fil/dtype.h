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
#include <triton/core/tritonserver.h>

template <TRITONSERVER_Datatype D>
struct TritonType {
};

template <typename T>
struct TritonDtype {
};

template <TRITONSERVER_TYPE_BOOL>
struct TritonType {
  typedef bool type;
};

template <bool>
struct TritonDtype {
  static constexpr TRITONSERVER_Datatype value = TRITONSERVER_TYPE_BOOL;
};

template <TRITONSERVER_TYPE_UINT8>
struct TritonType {
  typedef uint8_t type;
};

template <uint8_t>
struct TritonDtype {
  static constexpr TRITONSERVER_Datatype value = TRITONSERVER_TYPE_UINT8;
};

template <TRITONSERVER_TYPE_UINT16>
struct TritonType {
  typedef uint16_t type;
};

template <uint16_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT16;
};

template <TRITONSERVER_TYPE_UINT32>
struct TritonType {
  typedef uint32_t type;
};

template <uint32_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT32;
};

template <TRITONSERVER_TYPE_UINT64>
struct TritonType {
  typedef uint64_t type;
};

template <uint64_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT64;
};

template <TRITONSERVER_TYPE_INT8>
struct TritonType {
  typedef int8_t type;
};

template <int8_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT8;
};

template <TRITONSERVER_TYPE_INT16>
struct TritonType {
  typedef int16_t type;
};

template <int16_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT16;
};

template <TRITONSERVER_TYPE_INT32>
struct TritonType {
  typedef int32_t type;
};

template <int32_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT32;
};

template <TRITONSERVER_TYPE_INT64>
struct TritonType {
  typedef int64_t type;
};

template <int64_t>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT64;
};

template <TRITONSERVER_TYPE_FP32>
struct TritonType {
  typedef float type;
};

template <float>
struct TritonDtype {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_FP32;
};

template <TRITONSERVER_TYPE_FP64>
struct TritonType {
  typedef double type;
};
