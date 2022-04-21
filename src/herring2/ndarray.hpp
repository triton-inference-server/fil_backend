#pragma once
#include <functional>
#include <numeric>
#include <herring2/buffer.hpp>
#include <herring2/gpu_support.hpp>
#include <type_traits>

namespace herring {

template <typename T, index_t... layout>
struct ndarray {
  auto static constexpr N = sizeof...(layout);
  using index_type = index_t;

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  ndarray(buffer<T> const& in_buffer, Args&&... args)
    : data_{in_buffer},
      dims_{args...},
      cum_dims_ {[this]() {
        auto result = index_t[N]{};
        if constexpr (N > 0) {
          result[N - 1] = 1;
          for (auto i = N - 2; i > 0; --i) {
            result[i] = result[i + 1] * dims_[i];
          }
        }
        return result;
      }()}
  {
  }

  HOST DEVICE [[nodiscard]] auto* data() noexcept {
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

  auto get_index(std::array<std::size_t, N> indices) const {
    auto result = index_type{};
    for (auto i = index_type{}; i < N; ++i) {
      result += cum_dims_[layout[i]] * indices[i];
    }
    return result;
  }

  template<typename... Args, typename = typename std::enable_if_t<N == sizeof...(Args)>>
  auto get_index(Args&&... args) const {
    auto indices = std::array{args...};
    return get_index(indices);
  }

 private:
  buffer<T> data_;
  std::array<std::size_t, N> dims_;
  std::array<std::size_t, N> cum_dims_;
};

}
