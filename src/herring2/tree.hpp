#pragma once
#include <herring2/detail/index_type.hpp>
#include <herring2/gpu_support.hpp>
#include <herring2/tree_layout.hpp>

namespace herring {

enum class tree_layout {
  depth_first,
  breadth_first
};

/** A binary tree stored as a contiguous array of offsets to the most distant
 * child node*/
template <tree_layout layout, typename T>
struct tree {
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;

  HOST DEVICE tree(T* data, index_type node_count)
    : data_{data}, node_count_{node_count}
  {
  }

  HOST DEVICE [[nodiscard]] auto* data() noexcept { return data_; }
  HOST DEVICE [[nodiscard]] auto const* data() const noexcept { return data_; }

  HOST DEVICE [[nodiscard]] auto size() const noexcept { return node_count_; }

  HOST DEVICE [[nodiscard]] auto& operator[](index_type index) noexcept {
    return data()[index];
  }
  HOST DEVICE [[nodiscard]] auto const& operator[](index_type index) const noexcept {
    return data()[index];
  }

  HOST DEVICE [[nodiscard]] auto next_offset(index_type index, bool condition) {
    if constexpr (layout == tree_layout::depth_first) {
      return 1 + (data_[index] - 1) * condition;
    } else if constexpr (layout == tree_layout::breadth_first) {
      return data_[index] + condition - 1;
    } else {
      // static_assert(false);
    }
  }

 private:
  T* data_;
  raw_index_t node_count_;
};

}

