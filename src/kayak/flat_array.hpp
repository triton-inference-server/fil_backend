#pragma once
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>
#include <kayak/structured_data.hpp>
#include <kayak/structured_multidata.hpp>

namespace kayak {

using array_encoding = detail::array_encoding;

/** A 1D array of values */
template <array_encoding layout, typename T>
using flat_array = detail::flat_array<layout, T>;

template <array_encoding layout, typename T, bool bounds_check=false>
auto make_flat_array(
    typename flat_array<layout, T>::index_type size,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {
  return make_structured_data<flat_array<layout, T>, bounds_check>(
    size,
    mem_type,
    device,
    stream,
    size
  );
}

template<array_encoding layout, typename T>
using multi_flat_array = structured_multidata<flat_array<layout, T>>;

// TODO(wphicks): enable_if iter can be dereferenced to something convertible
// to index_type
template <array_encoding layout, typename T, typename iter>
auto make_multi_flat_array(
    iter sizes_begin,
    iter sizes_end,
    std::optional<detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>> align_to_bytes=std::nullopt,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {

  using obj_type = flat_array<layout, T>;
  using value_type = typename flat_array<layout, T>::value_type;

  auto padded_sizes = detail::get_padded_sizes<value_type>(
    sizes_begin, sizes_end, align_to_bytes
  );
  auto buffer_size = std::reduce(std::begin(padded_sizes), std::end(padded_sizes));

  auto data = buffer<value_type>(buffer_size, mem_type, device, stream);

  auto objs = std::vector<obj_type>{};
  objs.reserve(padded_sizes.size());

  auto offset = std::size_t{};
  auto obj_index = raw_index_t{};
  // TODO(wphicks): Avoid for_each
  std::for_each(
    sizes_begin,
    sizes_end,
    [&data, &objs, &padded_sizes, &offset, &obj_index](auto&& unpadded_size){
      objs.emplace_back(data.data() + offset, unpadded_size),
      offset += padded_sizes[obj_index];
      ++obj_index;
    }
  );

  return multi_flat_array<layout, T>{
    std::move(data), std::move(objs)
  };
}

}
