#pragma once
#include <cstddef>
#include <herring2/gpu_support.hpp>

namespace herring {
template<typename
struct buffer {
  using size_type = std::size_t;
  using value_type = T;

  using h_buffer       = T*;
  using d_buffer       = T*;
  using owned_h_buffer = std::unique_ptr<T[]>;
  using owned_d_buffer = detail::owned_device_buffer<T, IS_GPU_BUILD>;
  using data_store = std::variant<h_buffer, d_buffer, owned_h_buffer, owned_d_buffer>;
};

}
