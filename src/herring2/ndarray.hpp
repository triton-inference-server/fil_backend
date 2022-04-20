#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <herring2/buffer.hpp>
#include <type_traits>

namespace herring {

template <std::size_t N>
struct axis_index {
  constexpr axis_index(std::size_t index)
    : index_{[index]() constexpr {
      static_assert(index < N);
    }()} {}
 private:
  std::size_t index_;
};

/** Keep track of the order of axes in a particular data layout */
template <std::size_t N>
struct axis_order {
  // TODO(wphicks): Possible to compile-time enforce uniqueness of axes?
  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args), void>>
  constexpr axis_order(Args&&... args) : order_{args...} {
  }

  auto constexpr order() const {
    return order_;
  }

 private:
  std::array<axis_index<N>, N> order_;
};

template <bool bounds_check, std::size_t N, axis_order<N> layout, typename T>
struct ndarray {
  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  ndarray(buffer<T> const& in_buffer, Args&&... args)
    : data_{in_buffer},
      dims_{args...},
      cum_dims_ {[this]() {
        auto result = std::array<std::size_t, N>{};
        if constexpr (N > 0) {
          result[N - 1] = 1;
          for (auto i = N - 2; i > 0; --i) {
            result[i] = result[i + 1] * dims_[i];
          }
        }
        return result;
      }()}
  {
    if constexpr (bounds_check) {
      if (std::reduce(std::begin(dims_), std::end(dims_), std::size_t{1}, std::multiplies<>()) != data_.size()) {
        throw out_of_bounds("Array dimensions inconsistent with size of data buffer");
      }
    }
  }

  [[nodiscard]] auto* data() noexcept {
    return data_.get();
  }

  [[nodiscard]] auto& get_buffer() noexcept {
    return data_;
  }

  [[nodiscard]] auto size() const noexcept {
    return data_.size();
  }

  [[nodiscard]] auto const& dims() const noexcept {
    return dims_;
  }

  auto get_index(std::array<std::size_t, N> indices) const noexcept(!bounds_check) {
    auto result = std::size_t{};
    for (auto i = std::size_t{}; i < N; ++i) {
      result += cum_dims_[layout[i]] * indices[i];
    }
    if constexpr (bounds_check) {
      if (result >= size()) {
        throw out_of_bounds("Attempted to access out-of-bounds element of ndarray");
      }
    }
    return result;
  }

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  auto get_index(Args&&... args) const noexcept(!bounds_check) {
    auto indices = std::array<std::size_t, N>{args...};
    return get_index(indices);
  }

 private:
  buffer<T> data_;
  std::array<std::size_t, N> dims_;
  std::array<std::size_t, N> cum_dims_;
};

}
