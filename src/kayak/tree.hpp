#pragma once
#include <kayak/cuda_stream.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/structured_data.hpp>
#include <kayak/structured_multidata.hpp>
#include <kayak/tree_layout.hpp>

namespace kayak {

/** A binary tree stored as a contiguous array of offsets to the most distant
 * child node*/
template <tree_layout layout, typename T>
struct tree {
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;
  using value_type = T;

  HOST DEVICE tree()
    : data_{nullptr}, node_count_{}
  {
  }

  HOST DEVICE tree(value_type* data, index_type node_count)
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
    condition |= (data_[index] == 0);
    if constexpr (layout == tree_layout::depth_first) {
      return 1 + (data_[index] - 1) * condition;
    } else if constexpr (layout == tree_layout::breadth_first) {
      return data_[index] + condition - 1;
    } else {
      static_assert(layout == tree_layout::depth_first);
    }
  }

 private:
  value_type* data_;
  raw_index_t node_count_;
};

template <tree_layout layout, typename T, bool bounds_check=false>
auto make_tree(
    typename tree<layout, T>::index_type size,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {
  return make_structured_data<tree<layout, T>, bounds_check>(
    size,
    mem_type,
    device,
    stream,
    size
  );
}

template<tree_layout layout, typename T>
using multi_tree = structured_multidata<tree<layout, T>>;

// TODO(wphicks): enable_if iter can be dereferenced to something convertible
// to index_type
template <tree_layout layout, typename T, typename iter>
auto make_multi_tree(
    iter counts_begin,
    iter counts_end,
    std::optional<detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>> align_to_bytes=std::nullopt,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {

  using obj_type = tree<layout, T>;
  using value_type = typename obj_type::value_type;

  auto padded_sizes = detail::get_padded_sizes<value_type>(
    counts_begin, counts_end, align_to_bytes
  );
  auto buffer_size = std::reduce(std::begin(padded_sizes), std::end(padded_sizes));

  auto data = buffer<value_type>{buffer_size, mem_type, device, stream};

  auto objs = buffer<obj_type>{padded_sizes.size()};

  auto offset = std::size_t{};
  auto obj_index = raw_index_t{};
  std::for_each(
    counts_begin,
    counts_end,
    [&data, &objs, &padded_sizes, &offset, &obj_index](auto&& unpadded_size){
      objs.data()[obj_index] = obj_type{data.data() + offset, unpadded_size};
      offset += padded_sizes[obj_index];
      ++obj_index;
    }
  );

  return multi_tree<layout, value_type>{
    std::move(data),
    buffer<obj_type>{std::move(objs), mem_type, device, stream}
  };
}

}

