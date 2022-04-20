#pragma once
#include <herring2/device_id.hpp>
#include <herring2/device_type.hpp>
#include <herring2/detail/owning_buffer/base.hpp>
#include <memory>
#include <type_traits>

namespace herring {
namespace detail {
template<typename T>
struct owning_buffer<device_type::cpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;

  owning_buffer(std::size_t size)
    : data_{std::make_unique<T[]>(size)}
  {
  }

  auto* get() const { return data_.get(); }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  std::unique_ptr<T[]> data_;
};
}
}

