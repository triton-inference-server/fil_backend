#pragma once
#include <stddef.h>
#include <stdint.h>
#include <herring2/buffer.hpp>
#include <herring2/exceptions.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {

enum class data_layout {
  dense_row_major,
  dense_col_major
};

/** A 2D array of values */
template <data_layout layout, typename T>
struct data_array {
  data_array(buffer<T> const& in_buffer, size_t row_count, size_t col_count)
    : data_{in_buffer},
      count_{ [row_count, col_count](){
        if constexpr (layout == data_layout::dense_row_major) {
          return col_count;
        } else if constexpr (layout == data_layout::dense_col_major) {
          return row_count;
        } else {
          // TODO(whicks) static_assert(false);
        }
      }()}
  {
  }

  data_array(buffer<T> const& in_buffer, size_t count)
    : data_{in_buffer}, count_{count}
  {
  }

  [[nodiscard]] auto* data() noexcept {
    return data_.data();
  }

  [[nodiscard]] auto& get_buffer() noexcept {
    return data_;
  }

  [[nodiscard]] auto size() const noexcept {
    return data_.size();
  }

  [[nodiscard]] auto get_row_count() const {
    if constexpr (layout == data_layout::dense_row_major) {
      return data_.size() / count_;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return count_;
    } else {
      // TODO(whicks) static_assert(false);
    }
  }

  [[nodiscard]] auto get_col_count() const {
    if constexpr (layout == data_layout::dense_row_major) {
      return count_;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return data_.size() / count_;
    } else {
      // TODO(whicks) static_assert(false);
    }
  }

  template<typename index_type>
  HOST DEVICE auto get_index(index_type row, index_type col) const noexcept {
    auto result = index_type{};
    if constexpr (layout == data_layout::dense_row_major) {
      result = count_ * row + col;
    } else if constexpr (layout == data_layout::dense_col_major) {
      result = count_ * col + row;
    } else {
      // TODO(whicks) static_assert(false);
    }
    return result;
  }

  template<typename index_type>
  HOST DEVICE auto get_value(index_type row, index_type col) const noexcept {
    return data()[get_index(row, col)];
  }

  template<typename index_type>
  HOST DEVICE auto& operator[](index_type index) noexcept {
    return data()[index];
  }

  template<typename index_type>
  HOST DEVICE auto const& operator[](index_type index) const noexcept {
    return data()[index];
  }

 private:
  buffer<T> data_;
  size_t count_;
};

}
