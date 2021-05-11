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

namespace triton { namespace backend { namespace fil {

template <typename T>
class optional {  // C++17: Switch to std::optional
 public:
  optional() : empty_{}, has_value_{false} {}

  optional(const T& input_value) : value_{input_value}, has_value_{true} {}
  optional(optional<T>&& other) : has_value_{other}
  {
    if (other) {
      value_ = *other;
    }
  }

  ~optional()
  {
    if (has_value_) {
      value_.~T();
    } else {
      empty_.~empty_byte();
    }
  }

  explicit operator bool() const { return has_value_; }
  T& operator*() { return value_; }
  T* operator->() { return &value_; }
  optional& operator=(const T& new_value)
  {
    value_ = new_value;
    has_value_ = true;
    return *this;
  }

 private:
  struct empty_byte {
  };
  union {
    empty_byte empty_;
    T value_;
  };
  bool has_value_;
};

}}}  // namespace triton::backend::fil
