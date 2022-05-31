/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <omp.h>

template <typename T>
struct thread_count {
  thread_count() : value{omp_get_max_threads()} {}
  thread_count(T t) : value{
    [](T t) {
      auto result = T{t};
      auto max_count = omp_get_max_threads();
      if ( t < 1 || t > max_count) {
        result = max_count;
      }
      return result;
    }(t)} {}
  operator int() const { return static_cast<int>(value); }
 private:
  T value;
};
