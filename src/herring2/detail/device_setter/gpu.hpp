#pragma once
#include <cuda_runtime_api.h>
#include <herring2/cuda_check.hpp>
#include <herring2/detail/device_setter/base.hpp>
#include <herring2/device_type.hpp>
#include <herring2/device_id.hpp>

namespace herring {
namespace detail {

/** Struct for setting current device within a code block */
template <>
struct device_setter<device_type::gpu> {
  device_setter(herring::device_id<device_type::gpu> device) noexcept(false) : prev_device_{} {
    herring::cuda_check(cudaSetDevice(device.value()));
  }

  ~device_setter() {
    cudaSetDevice(prev_device_.value());
  }
 private:
  device_id<device_type::gpu> prev_device_;
};

}
}
