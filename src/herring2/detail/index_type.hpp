#pragma once
#include <limits.h>
#include <stdint.h>
#include <stddef.h>

#include <exception>
#include <herring2/detail/host_only_throw.hpp>
#include <herring2/gpu_support.hpp>
#include <type_traits>

namespace herring {
namespace detail {

auto constexpr static const MAX_INDEX = UINT_MAX;
auto constexpr static const MAX_DIFF = INT_MAX;
auto constexpr static const MIN_DIFF = INT_MIN;

struct bad_index : std::exception {
  bad_index() : bad_index("Invalid index") {}
  bad_index(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template <typename T, typename U>
HOST DEVICE auto constexpr universal_max(T a, U b) {
  return (a > b) ? a : b;
}

template <typename T, typename U>
HOST DEVICE auto constexpr universal_min(T a, U b) {
  return (a < b) ? a : b;
}

template <bool bounds_check>
struct index_type {
  using value_type = uint32_t;
  HOST DEVICE index_type() : val{} {};
  HOST DEVICE index_type(value_type index): val{index} {}
  HOST DEVICE index_type(size_t index) noexcept(!bounds_check) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index > MAX_INDEX) {
        herring::host_only_throw<bad_index>("Index exceeds maximum allowed value");
      }
    }
    result = universal_min(index, MAX_INDEX);
    return result;
  }()} {}
  HOST DEVICE index_type(int index) noexcept(!bounds_check) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index < 0 || index > MAX_INDEX) {
        herring::host_only_throw<bad_index>("Invalid value for index");
      }
    }
    result = universal_min(static_cast<uint32_t>(universal_max(0, index)), MAX_INDEX);
    return result;
  }()} {}
  HOST DEVICE operator value_type&() noexcept { return val; }
  HOST DEVICE operator value_type() const noexcept { return val; }
  HOST DEVICE operator size_t() const noexcept { return static_cast<size_t>(val); }
  HOST DEVICE auto value() const { return val; }

 private:
  value_type val;
};

template <bool bounds_check>
struct diff_type {
  using value_type = int32_t;
  HOST DEVICE diff_type() : val{} {};
  HOST DEVICE diff_type(value_type index): val{index} {}
  HOST DEVICE diff_type(ptrdiff_t index) : val{[index]() {
    auto result = value_type{};
    if constexpr (bounds_check) {
      if (index < MIN_DIFF || index > MAX_DIFF) {
        herring::host_only_throw<bad_index>("Invalid value for diff");
      }
    }
    result = universal_min(universal_max(index, MIN_DIFF), MAX_DIFF);
    return result;
  }()} {}
  HOST DEVICE operator value_type&() noexcept { return val; }
  HOST DEVICE operator value_type() const noexcept { return val; }
  HOST DEVICE operator ptrdiff_t() const noexcept { return static_cast<ptrdiff_t>(val); }
  HOST DEVICE auto value() const { return val; }

 private:
  value_type val;
};

template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, size_t> || std::is_same_v<T, typename index_type<bounds_check>::value_type>, bool> operator==(index_type<bounds_check> const& lhs, T const& rhs) {
  return lhs.value() == rhs;
}
template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, size_t> || std::is_same_v<T, typename index_type<bounds_check>::value_type>, bool> operator==(T const& lhs, index_type<bounds_check> const& rhs) {
  return lhs == rhs.value();
}

template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, size_t> || std::is_same_v<T, typename index_type<bounds_check>::value_type>, bool> operator!=(index_type<bounds_check> const& lhs, T const& rhs) {
  return !(lhs == rhs);
}
template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, size_t> || std::is_same_v<T, typename index_type<bounds_check>::value_type>, bool> operator!=(T const& lhs, index_type<bounds_check> const& rhs) {
  return !(lhs == rhs);
}

template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, ptrdiff_t> || std::is_same_v<T, typename diff_type<bounds_check>::value_type>, bool> operator==(diff_type<bounds_check> const& lhs, T const& rhs) {
  return lhs.value() == rhs;
}
template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, ptrdiff_t> || std::is_same_v<T, typename diff_type<bounds_check>::value_type>, bool> operator==(T const& lhs, diff_type<bounds_check> const& rhs) {
  return lhs == rhs.value();
}

template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, ptrdiff_t> || std::is_same_v<T, typename diff_type<bounds_check>::value_type>, bool> operator!=(diff_type<bounds_check> const& lhs, T const& rhs) {
  return !(lhs == rhs);
}
template <bool bounds_check, typename T>
HOST DEVICE std::enable_if_t<std::is_same_v<T, ptrdiff_t> || std::is_same_v<T, typename diff_type<bounds_check>::value_type>, bool> operator!=(T const& lhs, diff_type<bounds_check> const& rhs) {
  return !(lhs == rhs);
}

}
}
