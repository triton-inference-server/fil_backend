#pragma once
#include <kayak/detail/index_type.hpp>
#include <kayak/gpu_support.hpp>

namespace kayak {

enum class array_encoding {
  dense
  // TODO(wphicks): sparse
};

/** A 2D array of values */
template <array_encoding layout, typename T>
struct flat_array {
  using value_type = T;
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;

  HOST DEVICE flat_array()
    : data_{nullptr}, size_{0}
  {
  }

  HOST DEVICE flat_array(T* data, index_type size)
    : data_{data}, size_{size}
  {
  }

  HOST DEVICE [[nodiscard]] auto* data() noexcept { return data_; }
  HOST DEVICE [[nodiscard]] auto const* data() const noexcept { return data_; }

  HOST DEVICE [[nodiscard]] auto size() const noexcept { return size_; }

  HOST DEVICE [[nodiscard]] auto const& at(index_type index) const noexcept {
    return data()[index];
  }

  HOST DEVICE [[nodiscard]] auto& at(index_type index) noexcept {
    return data()[index];
  }

  HOST DEVICE [[nodiscard]] auto& operator[](index_type index) noexcept {
    return data()[index];
  }
  HOST DEVICE [[nodiscard]] auto const& operator[](index_type index) const noexcept {
    return data()[index];
  }

 private:
  T* data_;
  raw_index_t size_;
};

}
