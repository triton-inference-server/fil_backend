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

#include <type_traits>

namespace herring {
  template<typename T, template<typename...> class U>
  struct is_container_specialization : std::false_type {
    using value_type = T;
  };

  template<template<typename...> class U, typename... Args>
  struct is_container_specialization<U<Args...>, U>: std::true_type {
    using value_type = typename U<Args...>::value_type;
  };
}
