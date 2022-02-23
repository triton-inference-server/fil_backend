#pragma once

#include <type_traits>

namespace herring {
  template<typename T, template<typename...> class U>
  struct is_specialization : std::false_type {};

  template<template<typename...> class U, typename... Args>
  struct is_specialization<U<Args...>, U>: std::true_type {};
}
