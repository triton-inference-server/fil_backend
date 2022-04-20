#pragma once
#include <cstddef>
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
  data_array(buffer<T> const& in_buffer, std::size_t row_count, std::size_t col_count)
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

  data_array(buffer<T> const& in_buffer, std::size_t count)
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

  template<bool bounds_check, typename index_type>
  HOST DEVICE auto get_index(index_type row, index_type col) const noexcept(!bounds_check) {
    auto result = index_type{};
    if constexpr (layout == data_layout::dense_row_major) {
      result = count_ * row + col;
      if constexpr (bounds_check) {
        if (col > count_) {
          throw out_of_bounds("Column index exceeds number of columns");
        }
        if (result >= size()) {
          throw out_of_bounds("Row index exceeds number of rows");
        }
      }
    } else if constexpr (layout == data_layout::dense_col_major) {
      result = count_ * col + row;
      if constexpr (bounds_check) {
        if (row > count_) {
          throw out_of_bounds("Row index exceeds number of rows");
        }
        if (result >= size()) {
          throw out_of_bounds("Column index exceeds number of columns");
        }
      }
    } else {
      // TODO(whicks) static_assert(false);
    }
    return result;
  }

  template<bool bounds_check, typename index_type>
  HOST DEVICE auto get_value(index_type row, index_type col) const noexcept(!bounds_check) {
    return data()[get_index<bounds_check>(row, col)];
  }

 private:
  buffer<T> data_;
  std::size_t count_;
};

}
