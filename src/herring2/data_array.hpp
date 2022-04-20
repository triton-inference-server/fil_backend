#pragma once
#include <cstddef>
#include <herring2/buffer.hpp>

namespace herring {

enum class data_layout {
  dense_row_major,
  dense_col_major
};

template <data_layout layout, typename T>
struct data_array {

  auto* data() {
    return data_.data();
  }

  auto& get_buffer() {
    return data_;
  }

  auto size() {
    return data_.size();
  }

  auto get_row_count() const {
    if constexpr (layout == data_layout::dense_row_major) {
      return data_.size() / count;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return count;
    } else {
      // static_assert(false);
    }
  }

  auto get_col_count() const {
    if constexpr (layout == data_layout::dense_row_major) {
      return count;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return data_.size() / count;
    } else {
      // static_assert(false);
    }
  }

  auto get_index(std::size_t row, std::size_t col) const {
    if constexpr (layout == data_layout::dense_row_major) {
      return count * row + col;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return count * col + row;
    } else {
      // static_assert(false);
    }
  }

  auto get_value(std::size_t row, std::size_t col) const {
    return *(data_.data() + get_index(row, col));
  }

  void set_value(std::size_t row, std::size_t col, T value) {
    *(data_.data() + get_index(row, col)) = value;
  }

 private:
  buffer<T> data_;
  std::size_t count;
};

}
