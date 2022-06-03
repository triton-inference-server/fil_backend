#pragma once
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
  return uint32_t(result / 2);
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
  int device_id
) {
  // TODO (wphicks): I'm not sure if all of this tuning to get the optimal
  // shared memory configuration will actually be worthwhile, especially for
  // large batches. It may be better to just take the hit of writing to global
  // memory or using atomic adds rather than risk having configurations with
  // too many blocks or too few threads per block.

  // Chunk size is chosen such that each warp reads from memory aligned to the
  // size of a cache line
  auto chunks = kayak::padded_size(in.rows(), gpu::CHUNK_SIZE) / gpu::CHUNK_SIZE;
  std::cout << "PADDED SIZE of " << in.rows() << " is " << kayak::padded_size(in.rows(), gpu::CHUNK_SIZE) << "\n";
  std::cout << "Processing " << in.rows() << " rows as " << chunks << " chunks\n";
  auto sm_count = get_sm_count(device_id);
  std::cout << "Processing on " << sm_count << " streaming multiprocessors\n";
  auto max_shared_mem_entries_per_block = (
    get_max_shared_mem_per_block(device_id) /
    sizeof(io_t)
  );
  std::cout << "GPU can store " << max_shared_mem_entries_per_block << " entries per block in  " << get_max_shared_mem_per_block(device_id) << " bytes of shared memory\n";

  // Need at least enough memory to handle output for a single warp
  if (max_shared_mem_entries_per_block < num_class * gpu::WARP_SIZE) {
    // TODO(wphicks): Allocate global workspace memory rather than throwing
    throw unusable_model_exception(
      "Too many classes for available shared memory"
    );
  }

  // Each block processes a warp's worth of chunks per pass, and we wish to
  // minimize the number of passes, so ideally we would have as many blocks as
  // there are chunks divided by the size of a warp.
  auto blocks = kayak::padded_size(chunks, gpu::WARP_SIZE) / gpu::WARP_SIZE;
  if (blocks > sm_count) {
    blocks = kayak::downpadded_size(blocks, sm_count);
  }
  auto chunks_per_block = kayak::padded_size(chunks, blocks) / blocks;
  auto rows_per_block = chunks_per_block * gpu::CHUNK_SIZE;

  auto threads_per_block = kayak::downpadded_size(
    max_shared_mem_entries_per_block / (rows_per_block * num_class),
    gpu::WARP_SIZE
  );
  std::cout << "Ideal threads per block: " << threads_per_block << "\n";
  auto max_threads_per_block = get_max_threads_per_block(device_id);
  threads_per_block = kayak::detail::universal_min(threads_per_block, max_threads_per_block);

  if (threads_per_block == 0u) {
    std::cout << "Recomputing TPB for worst case...\n";
    // In this case, we are using too much shared memory per block, so we must
    // increase the number of blocks and process fewer chunks on each block. We
    // will compute the minimum number of blocks required to bring the shared
    // memory under the hardware limit.
    threads_per_block = gpu::WARP_SIZE;
    auto min_smem = threads_per_block * num_class;
    // We know that this will not be zero, since we already checked if we have
    // enough memory for the minimum memory consumption configuration
    rows_per_block = kayak::downpadded_size(
      max_shared_mem_entries_per_block / min_smem, gpu::CHUNK_SIZE
    );
    chunks_per_block = rows_per_block / gpu::CHUNK_SIZE;
    blocks = (
      chunks / chunks_per_block +
      kayak::raw_index_t{chunks % chunks_per_block == 0}
    );
  }
  std::cout << "Distributing " << in.rows() << " rows as " << chunks << " chunks over " << blocks << " blocks\n";

  std::cout << "Each block will process " << rows_per_block << " rows or " << chunks_per_block << " chunks\n";

  // Ensure that we have a minimum of one warp and a maximum no larger than the
  // maximum allowed per block or the maximum to fully occupy all SMs
  std::cout << "Current TPB: " << threads_per_block << "\n";
  std::cout << "Max TPB: " << get_max_threads_per_block(device_id) << "\n";
  std::cout << "Max threads: " << get_max_threads(device_id) << "\n";
  threads_per_block = kayak::detail::universal_max(
    kayak::detail::universal_min(
      kayak::detail::universal_min(threads_per_block, get_max_threads_per_block(device_id)),
      get_max_threads(device_id) / blocks
    ),
    gpu::WARP_SIZE
  );
  auto shared_memory_per_block = (
    rows_per_block * num_class * threads_per_block * sizeof(io_t)
  );
  std::cout << "Each block will have enough smem for " << rows_per_block * num_class * threads_per_block << " entries to accommodate " << rows_per_block << " rows of " << num_class << " classes across " << threads_per_block << " threads\n";
  auto result = kernel_params_t{};
  result.blocks = blocks;
  result.threads_per_block = threads_per_block;
  result.shared_memory_bytes_per_block = shared_memory_per_block;

  return result;
}

}
}
}
