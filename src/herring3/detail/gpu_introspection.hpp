#pragma once
#include <stddef.h>
#include <cuda_runtime_api.h>
#include <kayak/cuda_check.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
namespace herring {
namespace detail {

inline auto get_max_shared_mem_per_block(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_sm_count(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMultiProcessorCount,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_max_threads_per_block(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerBlock,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_max_threads_per_sm(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerMultiProcessor,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_max_shared_mem_per_sm(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_mem_clock_rate(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMemoryClockRate,
      device_id.value()
    )
  );
  return size_t(result);
}

inline auto get_core_clock_rate(kayak::device_id<kayak::device_type::gpu> device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrClockRate,
      device_id.value()
    )
  );
  return size_t(result);
}

template <typename T>
auto get_max_active_blocks_per_sm(
  T kernel, std::size_t block_size, std::size_t dynamic_smem_size=std::size_t{}
) {
  auto result = int{};
  kayak::cuda_check(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &result, kernel, block_size, dynamic_smem_size
    )
  );
  return size_t(result);
}

auto constexpr static const MAX_READ_CHUNK = size_t{128};
auto constexpr static const MAX_BLOCKS = size_t{65536};
auto constexpr static const WARP_SIZE = size_t{32};

}
}
