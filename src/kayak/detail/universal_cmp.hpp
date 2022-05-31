#pragma once
#include <kayak/gpu_support.hpp>

namespace kayak {
namespace detail {
template <typename T, typename U>
HOST DEVICE auto constexpr universal_max(T a, U b) {
  return (a > b) ? a : b;
}

template <typename T, typename U>
HOST DEVICE auto constexpr universal_min(T a, U b) {
  return (a < b) ? a : b;
}

}
}
