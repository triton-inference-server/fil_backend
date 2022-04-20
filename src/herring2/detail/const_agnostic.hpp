#pragma once
#include <type_traits>

namespace herring {
template <typename T, typename U, typename V = void>
using const_agnostic_same_t =
  std::enable_if_t<std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>, V>;
}
