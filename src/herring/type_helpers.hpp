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
