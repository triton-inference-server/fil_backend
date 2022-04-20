#pragma once
#include <herring2/detail/cuda_check/base.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/cuda_check/gpu.hpp>
#endif
#include <herring2/device_type.hpp>

namespace herring {
template <typename error_t>
void cuda_check(error_t const& err) noexcept(false) {
  detail::cuda_check<device_type::gpu>(err);
}
}
