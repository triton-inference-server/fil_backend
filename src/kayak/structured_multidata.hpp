#pragma once
#include <algorithm>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/detail/flat_array.hpp>
#include <kayak/exceptions.hpp>
#include <kayak/gpu_support.hpp>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace kayak {

template<typename struct_t>
struct structured_multidata {
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;
  using value_type = std::remove_reference_t<decltype(*std::declval<struct_t>().data())>;
  using obj_type = struct_t;

  structured_multidata() : data_{}, objs_{} { }

  structured_multidata(kayak::buffer<value_type>&& data, kayak::buffer<obj_type>&& objs)
    : data_{std::move(data)}, objs_{std::move(objs)} { }

  auto& buffer() { return data_; }
  auto const& buffer() const { return data_; }
  auto size() const { return objs_.size(); }
  auto buffer_size() const { return data_.size(); }
  auto bytes_size() const { return buffer_size() * sizeof(value_type); }

  template<bool bounds_check=false>
  structured_multidata(
    structured_multidata<obj_type> const& other,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) : data_{other.buffer().size(), mem_type, device, stream},
      objs_{other.objs_.size(), mem_type, device, stream} {
    copy<bounds_check>(data_, other.buffer());
    copy<bounds_check>(objs_, other.objs_);
  }

  auto objs() {
    return detail::flat_array<detail::array_encoding::dense, obj_type>{
      objs_.data(), objs_.size()
    };
  }

  auto& obj(index_type obj_index) {
    if (objs_.memory_type() != device_type::cpu) {
      throw wrong_device_type{
        "Attempted to retrieve host object from device memory"
      };
    }
    return objs_.data()[obj_index];
  }

  auto const& obj(index_type obj_index) const {
    if (objs_.mem_type() != device_type::cpu) {
      throw wrong_device_type{
        "Attempted to retrieve host object from device memory"
      };
    }
    return objs_.data()[obj_index];
  }

 private:
  kayak::buffer<value_type> data_;
  kayak::buffer<obj_type> objs_;
};

namespace detail {

template<typename T, typename iter>
auto get_padded_sizes(
  iter sizes_begin,
  iter sizes_end,
  detail::index_type<!GPU_ENABLED && DEBUG_ENABLED> align_to_bytes
) {
  auto padded_sizes = std::vector<raw_index_t>{};
  padded_sizes.reserve(std::distance(sizes_begin, sizes_end));
  std::transform(
    sizes_begin,
    sizes_end,
    std::back_inserter(padded_sizes),
    [&align_to_bytes](auto&& size) {
      auto result = raw_index_t{};
      if (align_to_bytes == 0) {
        result = size;
      } else {
        result = (
          std::lcm(align_to_bytes.value(), sizeof(T))
          - size * sizeof(T)
        ) / sizeof(T) + size;
      }
      return result;
    }
  );
  return padded_sizes;
}

}

}
