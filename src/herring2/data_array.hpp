#pragma once
#include <cstddef>

namespace herring {

enum class data_layout {
  dense_row_major,
  dense_col_major
};

template <data_layout layout, typename T>
struct data_array {
  T* buffer;
  std::size_t count;
  std::size_t size;

  auto get_row_count() {
    if constexpr (layout == data_layout::dense_row_major) {
      return size / count;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return count;
    } else {
      static_assert(false);
    }
  }

  auto get_col_count() {
    if constexpr (layout == data_layout::dense_row_major) {
      return count;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return size / count;
    } else {
      static_assert(false);
    }
  }

  auto get_index(std::size_t row, std::size_t feature) {
    if constexpr (layout == data_layout::dense_row_major) {
      return count * row + feature;
    } else if constexpr (layout == data_layout::dense_col_major) {
      return count * feature + row;
    } else {
      static_assert(false);
    }
  }

  auto get_value(std::size_t row, std::size_t feature) {
    return *(buffer + get_index(row, feature);
  }
};

}
