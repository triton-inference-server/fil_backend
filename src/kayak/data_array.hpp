#pragma once
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/structured_data.hpp>

namespace kayak {

enum class data_layout {
  dense_row_major,
  dense_col_major
  // TODO(wphicks): CSR and COO
};

/** A 2D array of values */
template <data_layout layout, typename T>
struct data_array {
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;

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

 private:
  T* data_;
  raw_index_t rows_;
  raw_index_t cols_;
};

template <data_layout layout, typename T, bool bounds_check=false>
auto make_data_array(
    typename data_array<layout, T>::index_type rows,
    typename data_array<layout, T>::index_type cols,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {
  return make_structured_data<data_array<layout, T>, bounds_check>(
    rows.value() * cols.value(),
    mem_type,
    device,
    stream,
    rows,
    cols
  );
}

}
