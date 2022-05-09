#pragma once
#include <stdint.h>
#include <type_traits>
#include <variant>
#include <kayak/detail/index_type.hpp>
#include <kayak/detail/raw_array.hpp>
#include <kayak/detail/universal_cmp.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/structured_data.hpp>
#include <kayak/structured_multidata.hpp>

namespace kayak {
template<typename storage_t>
struct bitset {
  using storage_type = storage_t;
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;

  auto constexpr static const bin_width = raw_index_t{
    detail::index_type<false>{sizeof(storage_type) * 8}
  };

  HOST DEVICE bitset()
    : data_{nullptr}, num_bits_{0}
  {
  }

  HOST DEVICE bitset(storage_type* data, index_type size)
    : data_{data}, num_bits_{size}
  {
  }

  HOST DEVICE bitset(storage_type* data)
    : data_{data}, num_bits_{detail::index_type<false>{sizeof(storage_type) * 8}}
  {
  }

  HOST DEVICE auto size() const {
    return num_bits_;
  }
  HOST DEVICE auto bin_count() const {
    return num_bits_ / bin_width + (num_bits_ % bin_width != 0);
  }

  // Standard bit-wise mutators and accessor
  HOST DEVICE auto& set(index_type index) {
    data_[bin_from_index(index)] |= mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto& clear(index_type index) {
    data_[bin_from_index(index)] &= ~mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto test(index_type index) const {
    auto result = false;
    if (index < num_bits_) {
      result = ((data_[bin_from_index(index)] & mask_in_bin(index)) != 0);
    }
    return result;
  }
  HOST DEVICE auto& flip() {
    for (auto i = raw_index_t{}; i < bin_count(); ++i) {
      data_[i] = ~data_[i];
    }
    return *this;
  }

  // Bit-wise boolean operations
  HOST DEVICE auto& operator&=(bitset<storage_type> const& other) {
    for (auto i = raw_index_t{}; i < detail::universal_min(size(), other.size()); ++i) {
      data_[i] &= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator|=(bitset<storage_t> const& other) {
    for (auto i = raw_index_t{}; i < detail::universal_min(size(), other.size()); ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator^=(bitset<storage_t> const& other) {
    for (auto i = raw_index_t{}; i < detail::universal_min(size(), other.size()); ++i) {
      data_[i] ^= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator~() const {
    flip();
    return *this;
  }

 private:
  storage_type* data_;
  raw_index_t num_bits_;

  HOST DEVICE auto mask_in_bin(raw_index_t index) const {
    return storage_t{1} << (index % bin_width);
  }

  HOST DEVICE auto bin_from_index(raw_index_t index) const {
    return index / bin_width;
  }
};

template <typename storage_t, bool bounds_check=false>
auto make_bitset(
    typename bitset<storage_t>::index_type size,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {
  return make_structured_data<bitset<storage_t>, bounds_check>(
    size,
    mem_type,
    device,
    stream,
    size
  );
}

template<typename T>
using multi_bitset = structured_multidata<bitset<T>>;

// TODO(wphicks): enable_if iter can be dereferenced to something convertible
// to index_type
template <typename T, typename iter>
auto make_multi_bitset(
    iter sizes_begin,
    iter sizes_end,
    std::optional<detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>> align_to_bytes=std::nullopt,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) {

  using obj_type = bitset<T>;
  using value_type = T;

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

  return multi_bitset<T>{
    std::move(data), std::move(objs)
  };
}

}
