#pragma once
#include <herring2/cuda_check.hpp>
#include <herring2/detail/device_id/base.hpp>
#include <herring2/device_type.hpp>
#include <rmm/cuda_device.hpp>

namespace herring {
namespace detail {
template<>
struct device_id<device_type::gpu> {
  using value_type = typename rmm::cuda_device_id::value_type;
  device_id() noexcept(false) : id_{[](){
    auto raw_id = value_type{};
    herring::cuda_check(cudaGetDevice(&raw_id));
    return raw_id;
  }()} {};
  device_id(value_type dev_id) noexcept : id_{dev_id} {};

  auto value() const noexcept { return id_.value(); }
 private:
  rmm::cuda_device_id id_;
};
}
}
