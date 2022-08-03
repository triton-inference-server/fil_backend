#pragma once
#include <stddef.h>
#include <cuda_runtime_api.h>
#include <kayak/cuda_check.hpp>
namespace herring {

inline auto get_max_shared_mem_per_block(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id
    )
  );
  return size_t(result);
}

inline auto get_sm_count(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMultiProcessorCount,
      device_id
    )
  );
  return size_t(result);
}

inline auto get_max_shared_mem_per_sm(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
      device_id
    )
  );
  return size_t(result);
}

inline auto get_mem_clock_rate(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMemoryClockRate,
      device_id
    )
  );
  return size_t(result);
}

inline auto get_core_clock_rate(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrClockRate,
      device_id
    )
  );
  return size_t(result);
}

auto constexpr static const MAX_RESIDENT_THREADS_PER_SM = size_t{2048};
auto constexpr static const MAX_READ_CHUNK = size_t{128};
auto constexpr static const MAX_BLOCKS = size_t{65536};

}
