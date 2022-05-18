#pragma once
#include <exception>
#include <kayak/gpu_support.hpp>
#include <kayak/detail/host_only_throw.hpp>
#include <kayak/detail/universal_cmp.hpp>
#include <limits>

namespace kayak {

struct invalid_narrowing : std::exception {
  invalid_narrowing() : invalid_narrowing("Invalid narrowing") {}
  invalid_narrowing(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template<bool bounds_check, typename to_t, typename from_t>
HOST DEVICE auto narrow_cast(from_t val) noexcept(!bounds_check) {
  if constexpr (bounds_check) {
    if (val > std::numeric_limits<to_t>::max() || val < std::numeric_limits<to_t>::min()) {
      host_only_throw<invalid_narrowing>();
    }
  }
  return to_t{
    detail::universal_min(
      detail::universal_max(val, std::numeric_limits<to_t>::min()),
      std::numeric_limits<to_t>::max()
    )
  };
}

}
