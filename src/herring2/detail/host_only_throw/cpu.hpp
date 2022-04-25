#pragma once
#include <herring2/detail/host_only_throw/base.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {
namespace detail {
template<typename T>
struct host_only_throw<T, true>{
  template <typename... Args>
  host_only_throw(Args&&... args) noexcept(false)  {
    throw T{std::forward<Args>(args)...};
  }
};
}
}
