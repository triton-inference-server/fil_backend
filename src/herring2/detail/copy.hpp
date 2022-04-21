#pragma once
#include <herring2/cuda_stream.hpp>
#include <herring2/detail/copy/cpu.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/copy/gpu.hpp>
#endif
#include <herring2/device_type.hpp>

namespace herring {
template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, std::size_t size, std::size_t dst_offset, std::size_t src_offset) {
  detail::copy<dst_type, src_type, T>(dst + dst_offset, src + src_offset, size, cuda_stream{});
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, std::size_t size, std::size_t dst_offset, std::size_t src_offset, cuda_stream stream) {
  detail::copy<dst_type, src_type, T>(dst + dst_offset, src + src_offset, size, stream);
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, std::size_t size) {
  detail::copy<dst_type, src_type, T>(dst, src, size, cuda_stream{});
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, std::size_t size, cuda_stream stream) {
  detail::copy<dst_type, src_type, T>(dst, src, size, stream);
}

template<typename T>
void copy(T* dst, T const* src, std::size_t size, device_type dst_type, device_type src_type, std::size_t dst_offset, std::size_t src_offset, cuda_stream stream) {
  if (dst_type == device_type::gpu && src_type == device_type::gpu) {
    detail::copy<device_type::gpu, device_type::gpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::cpu && src_type == device_type::cpu) {
    detail::copy<device_type::cpu, device_type::cpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::gpu && src_type == device_type::cpu) {
    detail::copy<device_type::gpu, device_type::cpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::cpu && src_type == device_type::gpu) {
    detail::copy<device_type::cpu, device_type::gpu, T>(dst + dst_offset, src + src_offset, size, stream);
  }
}

template<typename T>
void copy(T* dst, T const* src, std::size_t size, device_type dst_type, device_type src_type) {
  copy<T>(dst, src, size, dst_type, src_type, 0, 0, cuda_stream{});
}

template<typename T>
void copy(T* dst, T const* src, std::size_t size, device_type dst_type, device_type src_type, cuda_stream stream) {
  copy<T>(dst, src, size, dst_type, src_type, 0, 0, stream);
}

}
