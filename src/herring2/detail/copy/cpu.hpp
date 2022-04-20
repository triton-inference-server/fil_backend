#pragma once
#include <cstring>
#include <herring2/detail/cuda_stream.hpp>
#include <herring2/device_type.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {
namespace detail {

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<dst_type == device_type::cpu && src_type == device_type::cpu, void> copy(T* dst, T const* src, std::size_t size, cuda_stream stream) {
  std::memcpy(dst, src, size * sizeof(T));
}

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<(dst_type != device_type::cpu || src_type != device_type::cpu) && !GPU_ENABLED, void> copy(T* dst, T const* src, std::size_t size, cuda_stream stream) {
  throw gpu_unsupported("Copying from or to device in non-GPU build");
}

}
}
