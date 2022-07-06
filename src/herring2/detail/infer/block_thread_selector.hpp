#pragma once
#include <nvtx3/nvtx3.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <herring2/detail/gpu_constants.hpp>
#include <herring2/exceptions.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/detail/universal_cmp.hpp>
#include <kayak/padding.hpp>

namespace herring {
namespace detail {
namespace inference {

inline auto get_max_shared_mem_per_block(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id
    )
  );
  return uint32_t(result);
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
  return uint32_t(result);
}

inline auto get_max_threads_per_sm(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerMultiProcessor,
      device_id
    )
  );
  return uint32_t(result);
}

inline auto get_max_threads(int device_id) {
  return get_max_threads_per_sm(device_id) * get_sm_count(device_id);
}

inline auto get_max_threads_per_block(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerBlock,
      device_id
    )
  );
  return uint32_t(result);
}

struct kernel_params_t{
  uint32_t blocks;
  uint32_t threads_per_block;
  size_t shared_memory_bytes_per_block;
};

template <typename io_t>
auto block_thread_selector(
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  kayak::raw_index_t num_class,
  kayak::raw_index_t num_trees,
  int device_id
) {
  NVTX3_FUNC_RANGE();

  // Must be able to process a warp's worth of chunks or the total number of
  // rows in a single block, whichever is smaller
  auto min_rows_per_block = kayak::detail::universal_min(
    in.rows(),
    gpu::WARP_SIZE * gpu::CHUNK_SIZE
  );

  auto max_smem_entries = get_max_shared_mem_per_block(device_id) / sizeof(io_t{});
  std::cout << "Max shared mem per block: " << get_max_shared_mem_per_block(device_id) << "\n";
  std::cout << "Max threads per block: " << get_max_threads_per_block(device_id) << "\n";

  // Each warp writes to a different memory location for each row that it
  // processes. Having more warps means that we require more shared memory
  // because we require more of these unique memory locations.
  //
  // Here, we use as much shared memory as possible without exceeding the
  // number of trees, since any warps in excess of the number of trees will be
  // wasted.
  auto max_warps_per_block = kayak::detail::universal_min(
    num_trees,
    max_smem_entries / (min_rows_per_block * num_class)
  );

  if (max_warps_per_block == 0) {
    // TODO: Allocate a global workspace instead
    throw unusable_model_exception("Too many classes for available shared memory");
  }

  // Make sure that we have not exceeded hardware limits on the number of warps
  auto warps_per_block = kayak::detail::universal_min(
    max_warps_per_block,
    get_max_threads_per_block(device_id) / gpu::WARP_SIZE
  );

  // Now that we have fixed the number of warps per block, we determine the
  // maximum number of chunks we can process on each block based on the available
  // shared memory. Every warp writes to a unique location for each row that it
  // processes, so we divide available memory by the amount of memory required
  // for each warp to store the output for a single chunk.
  auto max_chunks_per_block = max_smem_entries / (gpu::CHUNK_SIZE * num_class * warps_per_block);

  if (max_chunks_per_block * gpu::CHUNK_SIZE < min_rows_per_block ) {
    throw unusable_model_exception("Unexpected error calculating chunks per block");
  }

  auto used_smem_entries = max_chunks_per_block * gpu::CHUNK_SIZE * num_class * warps_per_block;

  auto total_chunks = kayak::padded_size(in.rows(), gpu::CHUNK_SIZE) / gpu::CHUNK_SIZE;

  // Now we just need enough blocks to process all chunks
  auto total_blocks = kayak::padded_size(total_chunks, max_chunks_per_block) / max_chunks_per_block;

  if (total_blocks == 0) {
    throw unusable_model_exception("Unexpected error calculating total blocks");
  }

  auto result = kernel_params_t{};

  result.blocks = total_blocks;
  result.threads_per_block = warps_per_block * gpu::WARP_SIZE;
  result.shared_memory_bytes_per_block = used_smem_entries * sizeof(io_t{});

  return result;
}

}
}
}
