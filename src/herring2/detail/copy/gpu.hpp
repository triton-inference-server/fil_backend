#pragma once
#include <cuda_runtime_api.h>
#include <herring2/cuda_check.hpp>
#include <herring2/detail/cuda_stream.hpp>
#include <herring2/gpu_support.hpp>
#include <type_traits>

namespace herring {
namespace detail {

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<(dst_type == device_type::gpu || src_type == device_type::gpu) && GPU_ENABLED, void> copy(T* dst, T const* src, std::size_t size, cuda_stream stream) {
  cuda_check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
}

}
}
