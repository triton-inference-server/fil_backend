#pragma once
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>

namespace kayak {

enum class data_layout {
  dense_row_major,
  dense_col_major
  // TODO(wphicks): CSR and COO
};

/** A 2D array of values */
template <data_layout layout_v, typename T>
struct data_array {
  using value_type = T;
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;
  auto constexpr static const layout = layout_v;

  HOST DEVICE data_array(T* data, index_type row_count, index_type col_count)
    : data_{data}, rows_{row_count}, cols_{col_count}
  {
  }

  HOST DEVICE [[nodiscard]] auto* data() noexcept { return data_; }
  HOST DEVICE [[nodiscard]] auto const* data() const noexcept { return data_; }

  HOST DEVICE [[nodiscard]] auto size() const noexcept { return rows_ * cols_; }

  HOST DEVICE [[nodiscard]] auto rows() const { return rows_; }

  HOST DEVICE [[nodiscard]] auto cols() const { return cols_; }

  HOST DEVICE [[nodiscard]] auto get_index(index_type row, index_type col) const noexcept {
    auto result = index_type{};
    if constexpr (layout == data_layout::dense_row_major) {
      result = cols_ * row + col;
    } else if constexpr (layout == data_layout::dense_col_major) {
      result = rows_ * col + row;
    } else {
      static_assert(layout == data_layout::dense_row_major);
    }
    return result;
  }

  HOST DEVICE [[nodiscard]] auto const& at(index_type row, index_type col) const noexcept {
    return data()[get_index(row, col)];
  }

  HOST DEVICE [[nodiscard]] auto& at(index_type row, index_type col) noexcept {
    return data()[get_index(row, col)];
  }

  template<typename U>
  HOST DEVICE [[nodiscard]] auto& other_at(U* other_data, index_type row, index_type col) const noexcept {
    return other_data[get_index(row, col)];
  }

  HOST DEVICE [[nodiscard]] auto& operator[](index_type index) noexcept {
    return data()[index];
  }
  HOST DEVICE [[nodiscard]] auto const& operator[](index_type index) const noexcept {
    return data()[index];
  }

  template<typename lambda_t>
  HOST DEVICE void for_each(lambda_t&& lambda) {
    for (auto i=raw_index_t{}; i < rows_ * cols_; ++i) {
      lambda(data_[i]);
    }
  }

  template<typename lambda_t>
  HOST DEVICE void for_each_col(index_type row, lambda_t&& lambda) {
    if constexpr (layout == data_layout::dense_row_major) {
      auto begin = data_ + get_index(row, index_type{});
      for (auto i=raw_index_t{}; i < cols_; ++i) {
        lambda(begin[i]);
      }
    } else {
      for (auto i=raw_index_t{}; i < cols_; ++i) {
        lambda(at(row, i));
      }
    }
  }

  template<typename lambda_t>
  HOST DEVICE void for_each_row(index_type col, lambda_t&& lambda) {
    if constexpr (layout == data_layout::dense_col_major) {
      auto begin = data_ + get_index(index_type{}, col);
      for (auto i=raw_index_t{}; i < rows_; ++i) {
        lambda(begin[i]);
      }
    } else {
      for (auto i=raw_index_t{}; i < rows_; ++i) {
        lambda(at(i, col));
      }
    }
  }

 private:
  T* data_;
  raw_index_t rows_;
  raw_index_t cols_;
};

}
