#pragma once
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <type_traits>
#include <utility>

namespace kayak {

template<typename struct_t>
struct structured_data {
  using value_type = std::remove_reference_t<decltype(*std::declval<struct_t>().data())>;
  using obj_type = struct_t;

  structured_data() : data_{}, obj_{} { }

  auto& buffer() { return data_; }
  auto const& buffer() const { return data_; }
  auto& obj() { return obj_; }
  auto const& obj() const { return obj_; }

  template<bool bounds_check=false>
  structured_data(
    structured_data<obj_type> const& other,
    device_type mem_type=device_type::cpu,
    int device=0,
    cuda_stream stream=cuda_stream{}
  ) : data_{other.buffer().size(), mem_type, device, stream}, obj_{other.obj()} {
    if constexpr (bounds_check) {
      if (obj_.size() > data_.size()) {
        throw out_of_bounds{
          "Attempted to create object without a large enough backing buffer"
        };
      }
    }
    copy<false>(data_, other.buffer());
  }

  template<bool bounds_check, typename... args_t>
  static auto make_structured_data(
    detail::index_type<bounds_check> size,
    device_type mem_type,
    int device,
    cuda_stream stream,
    args_t&&... args
  ) {
    using value_type = std::remove_reference_t<decltype(*std::declval<struct_t>().data())>;
    auto data = kayak::buffer<value_type>{size, mem_type, device, stream};
    auto obj = struct_t(data.data(), std::forward<args_t...>(args)...);
    if constexpr (bounds_check) {
      if (obj.size() > data.size()) {
        throw out_of_bounds{
          "Attempted to create object without a large enough backing buffer"
        };
      }
    }
    return structured_data{
      std::move(data),
      std::move(obj)
    };
  }

 private:
  kayak::buffer<value_type> data_;
  obj_type obj_;

  structured_data(kayak::buffer<value_type>&& data, struct_t&& obj)
    : data_{std::move(data)}, obj_{std::move(obj)} { }
};

template<typename struct_t, bool bounds_check, typename... args_t>
auto make_structured_data(
  detail::index_type<bounds_check> size,
  device_type mem_type,
  int device,
  cuda_stream stream,
  args_t&&... args
) {
  return structured_data<struct_t>::template make_structured_data<bounds_check, args_t...>(
    size,
    mem_type,
    device,
    stream,
    std::forward<args_t...>(args)...
  );
}

template<typename struct_t, bool bounds_check, typename... args_t>
auto make_structured_host_data(
  detail::index_type<bounds_check> size,
  args_t&&... args
) {
  return make_structured_data<struct_t, bounds_check, args_t...>(
    size,
    device_type::cpu,
    0,
    cuda_stream{},
    std::forward<args_t...>(args)...
  );
}

}
