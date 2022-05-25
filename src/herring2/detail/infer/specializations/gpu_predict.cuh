#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <herring2/detail/gpu_constants.hpp>
#include <herring2/detail/infer/infer_kernel.cuh>
#include <herring2/detail/infer/gpu.hpp>
#include <herring2/exceptions.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <type_traits>

namespace herring {
namespace detail {
namespace inference {

auto get_max_shared_mem_per_block(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id
    )
  );
  return result;
}

auto get_sm_count(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMultiProcessorCount,
      device_id
    )
  );
  return result;
}

auto get_max_threads_per_sm(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerMultiProcessor,
      device_id
    )
  );
  return result;
}

auto get_max_threads(int device_id) {
  return get_max_threads_per_sm(device_id) * get_sm_count(device_id);
}

auto get_max_threads_per_block(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMaxThreadsPerBlock,
      device_id
    )
  );
  return result / WARP_SIZE;
}

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
>
std::enable_if_t<D == kayak::device_type::gpu && kayak::GPU_ENABLED, void> predict(
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  kayak::raw_index_t num_class,
  int device_id=0,
  kayak::cuda_stream stream = kayak::cuda_stream{}
) {
  // TODO (wphicks): I'm not sure if all of this tuning to get the optimal
  // shared memory configuration will actually be worthwhile, especially for
  // large batches. It may be better to just take the hit of writing to global
  // memory or using atomic adds rather than risk having configurations with
  // too many blocks or too few threads per block.

  // Chunk size is chosen such that each warp reads from memory aligned to the
  // size of a cache line
  auto chunks = padded_size(in.rows(), CHUNK_SIZE) / CHUNK_SIZE;
  std::cout << "Processing " << in.rows() << " rows as " << chunks << " chunks\n";
  auto sm_count = get_sm_count(device_id);
  std::cout << "Processing on " << sm_count << " streaming multiprocessors\n";
  auto max_shared_mem_entries_per_block = (
    get_max_shared_mem_per_block(device_id) /
    sizeof(typename forest_t::output_type)
  );
  std::cout << "GPU can store " << max_shared_mem_entries_per_block << " entries per block in shared memory\n";

  // Need at least enough memory to handle output for a single warp
  if (max_shared_mem_entries_per_block < num_class * WARP_SIZE) {
    // TODO(wphicks): Allocate global workspace memory rather than throwing
    throw unusable_model_exception(
      "Too many classes for available shared memory"
    );
  }

  // Each block processes a warp's worth of chunks per pass, and we wish to
  // minimize the number of passes, so ideally we would have as many blocks as
  // there are chunks divided by the size of a warp.
  auto blocks = padded_size(
    padded_size(chunks, WARP_SIZE) / WARP_SIZE,
    sm_count
  );
  auto chunks_per_block = padded_size(chunks, blocks) / blocks;
  auto rows_per_block = chunks_per_block * CHUNK_SIZE;

  auto threads_per_block = std::min(downpadded_size(
    max_shared_mem_entries_per_block / (rows_per_block * num_class),
    WARP_SIZE
  ), get_max_threads_per_block(device_id));

  if (threads_per_block == 0u) {
    // In this case, we are using too much shared memory per block, so we must
    // increase the number of blocks and process fewer chunks on each block. We
    // will compute the minimum number of blocks required to bring the shared
    // memory under the hardware limit.
    threads_per_block = WARP_SIZE;
    auto min_smem = threads_per_block * num_class;
    // We know that this will not be zero, since we already checked if we have
    // enough memory for the minimum memory consumption configuration
    rows_per_block = downpadded_size(
      max_shared_mem_entries_per_block / min_smem, CHUNK_SIZE
    );
    chunks_per_block = rows_per_block / CHUNK_SIZE;
    blocks = (
      chunks / chunks_per_block +
      raw_index_t{chunks % chunks_per_block == 0}
    );
  }
  std::cout << "Distributing " << in.rows() << " rows as " << chunks << " chunks over " << blocks << " blocks\n";

  std::cout << "Each block will process " << rows_per_block << " rows or " << chunks_per_block << " chunks\n";

  // Ensure that we have a minimum of one warp and a maximum no larger than the
  // maximum allowed per block or the maximum to fully occupy all SMs
  threads_per_block = std::max(
    std::min(
      std::min(threads_per_block, get_max_threads_per_block(device_id)),
      get_max_threads(device_id) / blocks
    ),
    WARP_SIZE
  );
  std::cout << forest.tree_count() << " trees will be distributed over " << threads_per_block / WARP_SIZE << " warps\n";

  auto shared_memory_per_block = (
    rows_per_block * num_class * threads_per_block * sizeof(typename forest_t::output_type)
  );

  auto categorical = forest.is_categorical();
  auto lookup = forest.requires_output_lookup();
  auto vector_leaf = forest.has_vector_leaves();
  auto algorithm_selector = (
    (kayak::raw_index_t{categorical} << 2u) +
    (kayak::raw_index_t{lookup} << 1u) +
    kayak::raw_index_t{vector_leaf}
  );

  switch(algorithm_selector) {
    case ((0u << 2) + (0u << 1) + 0u):
      infer_kernel<false, false, false><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((0u << 2) + (0u << 1) + 1u):
      infer_kernel<false, false, true><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((0u << 2) + (1u << 1) + 0u):
      infer_kernel<false, true, false><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((0u << 2) + (1u << 1) + 1u):
      infer_kernel<false, true, true><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((1u << 2) + (0u << 1) + 0u):
      infer_kernel<true, false, false><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((1u << 2) + (0u << 1) + 1u):
      infer_kernel<true, false, true><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    case ((1u << 2) + (1u << 1) + 0u):
      infer_kernel<true, true, true><<<blocks, threads_per_block, shared_memory_per_block, stream>>>(forest, out, in, num_class);
      break;
    default:
      throw unusable_model_exception("Unexpected algorithm selection");
  }
}

// single IO, single threshold, few features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, uint32_t, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, uint32_t, true>, float, float>;

// single IO, single threshold, many features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, uint32_t, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, uint32_t, true>, float, float>;

// single IO, double threshold, few features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, true>, float, float>;

// single IO, double threshold, many features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, uint32_t, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, float, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, float, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, double, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, double, true>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint32_t, false>, float, float>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint32_t, true>, float, float>;

// double IO, single threshold, few features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, uint32_t, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint16_t, uint32_t, uint32_t, uint32_t, true>, double, double>;

// double IO, single threshold, many features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint32_t, uint32_t, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, float, uint32_t, uint32_t, uint32_t, uint32_t, true>, double, double>;

// double IO, double threshold, few features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, true>, double, double>;

// double IO, double threshold, many features
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint16_t, uint32_t, uint32_t, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, float, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, float, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, double, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, double, true>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint32_t, false>, double, double>;
extern template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint32_t, uint32_t, uint32_t, uint32_t, true>, double, double>;

}
}
}

