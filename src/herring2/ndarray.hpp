#pragma once
#include <herring2/detail/index_type.hpp>
#include <herring2/gpu_support.hpp>
#include <type_traits>
#include <utility>

namespace herring {

namespace detail {
// Use raw arrays here for device-side support
template<typename T, raw_index_t N>
using raw_array = T[N];
}

template<typename... Axes>
auto nth_layout_elem(raw_index_t n, raw_index_t first, Axes const&... remaining) {
  auto result = raw_index_t{1};
  if (n == 0) {
    result = first;
  } else {
    if constexpr (sizeof...(remaining) != 0) {
      result = nth_layout_elem(n-1, remaining...);
    }
  }
  return result;
}

template <typename T, raw_index_t... layout>
struct ndarray {
  auto static constexpr const N = sizeof...(layout);

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  HOST DEVICE ndarray(T* data, Args&&... args)
    : data_{data},
      dims_{static_cast<raw_index_t>(static_cast<detail::index_type<false>>(args))...}
  {
    // TODO(wphicks): There is a more efficient way to calculate this
    for (auto i=raw_index_t{}; i < N; ++i) {
      auto cur_l = nth_layout_elem(i, layout...);
      cum_dims_[cur_l] = raw_index_t{1};
      for (auto j=i + 1; j < N; ++j) {
        cum_dims_[cur_l] *= dims_[nth_layout_elem(j, layout...)];
      }
    }
  }

  HOST DEVICE [[nodiscard]] auto const* data() const noexcept {
    return data_;
  }

  HOST DEVICE [[nodiscard]] auto* data() noexcept {
    return data_;
  }

  HOST DEVICE [[nodiscard]] auto size() const noexcept {
    auto result = raw_index_t{static_cast<int>(N != 0)};
    for (auto i = raw_index_t{}; i < N; ++i) {
      result *= dims_[i];
    }
    return result;
  }

  HOST DEVICE [[nodiscard]] auto constexpr axes() const noexcept {
    return detail::index_type<false>{N};
  }

  HOST DEVICE [[nodiscard]] auto const* dims() const noexcept {
    return dims_;
  }

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  HOST DEVICE [[nodiscard]] auto get_index(Args&&... args) const {
    return get_index(detail::raw_array<raw_index_t, N>{args...});
  }

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  HOST DEVICE [[nodiscard]] auto const& at(Args&&... args) const {
    return at(detail::raw_array<raw_index_t, N>{args...});
  }

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  HOST DEVICE [[nodiscard]] auto& at(Args&&... args) {
    return at(detail::raw_array<raw_index_t, N>{args...});
  }

  template<typename U, typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  HOST DEVICE [[nodiscard]] auto& other_at(U* other_data, Args&&... args) const {
    return other_at(other_data, detail::raw_array<raw_index_t, N>{args...});
  }

 private:
  T* data_;
  detail::raw_array<raw_index_t, N> dims_;
  detail::raw_array<raw_index_t, N> cum_dims_;

  HOST DEVICE [[nodiscard]] auto get_index(detail::raw_array<raw_index_t, N>&& indices) const {
    auto result = raw_index_t{};
    for (auto i = raw_index_t{}; i < N; ++i) {
      result += indices[i] * cum_dims_[i];
    }
    return result;
  }

  HOST DEVICE [[nodiscard]] auto const& at(detail::raw_array<raw_index_t, N>&& indices) const {
    return data()[get_index(indices)];
  }

  HOST DEVICE [[nodiscard]] auto& at(detail::raw_array<raw_index_t, N>&& indices) {
    return data()[get_index(std::move(indices))];
  }

  template <typename U>
  HOST DEVICE [[nodiscard]] auto& other_at(U* other_data, detail::raw_array<raw_index_t, N>&& indices) const {
    return other_data[get_index(indices)];
  }
};

}
