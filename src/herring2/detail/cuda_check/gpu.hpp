#pragma once
#include <cuda_runtime_api.h>
#include <herring2/detail/cuda_check/base.hpp>
#include <herring2/device_type.hpp>
#include <herring2/exceptions.hpp>
namespace herring {
namespace detail {

template <>
inline void cuda_check<device_type::gpu, cudaError_t>(cudaError_t const& err) noexcept(false) {
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw bad_cuda_call(cudaGetErrorString(err));
  }
}

}
}
