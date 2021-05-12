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

template <TRITONSERVER_DataType D>
struct TritonType {
};

template <typename T>
struct TritonDtype {
};

template <>
struct TritonType<TRITONSERVER_TYPE_BOOL> {
  typedef bool type;
};

template <>
struct TritonDtype<bool> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_BOOL;
};

template <>
struct TritonType<TRITONSERVER_TYPE_UINT8> {
  typedef uint8_t type;
};

template <>
struct TritonDtype<uint8_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT8;
};

template <>
struct TritonType<TRITONSERVER_TYPE_UINT16> {
  typedef uint16_t type;
};

template <>
struct TritonDtype<uint16_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT16;
};

template <>
struct TritonType<TRITONSERVER_TYPE_UINT32> {
  typedef uint32_t type;
};

template <>
struct TritonDtype<uint32_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT32;
};

template <>
struct TritonType<TRITONSERVER_TYPE_UINT64> {
  typedef uint64_t type;
};

template <>
struct TritonDtype<uint64_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_UINT64;
};

template <>
struct TritonType<TRITONSERVER_TYPE_INT8> {
  typedef int8_t type;
};

template <>
struct TritonDtype<int8_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT8;
};

template <>
struct TritonType<TRITONSERVER_TYPE_INT16> {
  typedef int16_t type;
};

template <>
struct TritonDtype<int16_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT16;
};

template <>
struct TritonType<TRITONSERVER_TYPE_INT32> {
  typedef int32_t type;
};

template <>
struct TritonDtype<int32_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT32;
};

template <>
struct TritonType<TRITONSERVER_TYPE_INT64> {
  typedef int64_t type;
};

template <>
struct TritonDtype<int64_t> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_INT64;
};

template <>
struct TritonType<TRITONSERVER_TYPE_FP32> {
  typedef float type;
};

template <>
struct TritonDtype<float> {
  static constexpr TRITONSERVER_DataType value = TRITONSERVER_TYPE_FP32;
};

template <>
struct TritonType<TRITONSERVER_TYPE_FP64> {
  typedef double type;
};
