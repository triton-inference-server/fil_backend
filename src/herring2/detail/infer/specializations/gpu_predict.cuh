#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <herring2/detail/gpu_constants.hpp>
#include <herring2/detail/infer/infer_kernel.cuh>
#include <herring2/detail/infer/gpu.hpp>
#include <herring2/exceptions.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_check.hpp>
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
  kayak::raw_index_t leaf_size,
  int device_id=0
) {
  auto categorical = forest.is_categorical();
  auto lookup = forest.requires_output_lookup();
  auto vector_leaf = forest.has_vector_leaves();
  auto algorithm_selector = (
    (kayak::raw_index_t{categorical} << 2u) +
    (kayak::raw_index_t{lookup} << 1u) +
    kayak::raw_index_t{vector_leaf}
  );

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

  // Need at least enough memory to handle output for a single chunk
  if (max_shared_mem_entries_per_block < num_class * CHUNK_SIZE) {
    // TODO(whicks): Allocate global workspace memory rather than throwing
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
  // If we can't fit enough output in shared memory for the number of rows per
  // block, we increase the number of blocks to reduce the required shared
  // memory per block.
  if(num_class * in.rows() > max_shared_mem_entries_per_block * blocks) {
    blocks = padded_size(
      num_class * in.rows() / (max_shared_mem_entries_per_block - num_class),
      sm_count
    );
  }
  std::cout << "Distributing " << in.rows() << " rows as " << chunks << " chunks over " << blocks << " blocks\n";
  // We want as many threads per block as possible without going over hardware
  // limits. Ideally, there will be one thread per chunk assigned to the block.
  auto chunks_per_block = padded_size(chunks, blocks) / blocks;
  auto rows_per_block = chunks_per_block * CHUNK_SIZE;
  std::cout << "Each block will process " << rows_per_block << " rows or " << chunks_per_block << " chunks\n";
  // Ensure that we have a minimum of one warp and a maximum no larger than the
  // maximum allowed per block or the maximum to fully occupy all SMs
  auto threads_per_block = std::max(
    std::min(
      std::min(chunks_per_block, get_max_threads_per_block(device_id)),
      get_max_threads(device_id) / blocks
    ),
    WARP_SIZE
  );
  std::cout << forest.tree_count() << " trees will be distributed over " << threads_per_block / WARP_SIZE << " warps\n";

  auto shared_memory_per_block = rows_per_block * num_class * sizeof(io_t);
  std::cout << "Each block will be given enough shared memory for " << rows_per_block * num_class << " entries to accommodate " << num_class << " class for " << rows_per_block " rows\n";

  // TODO: Stream
  switch(algorithm_selector) {
    case ((0u << 2) + (0u << 1) + 0u):
      infer_kernel<false, false, false><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((0u << 2) + (0u << 1) + 1u):
      infer_kernel<false, false, true><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((0u << 2) + (1u << 1) + 0u):
      infer_kernel<false, true, false><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((0u << 2) + (1u << 1) + 1u):
      infer_kernel<false, true, true><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((1u << 2) + (0u << 1) + 0u):
      infer_kernel<true, false, false><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((1u << 2) + (0u << 1) + 1u):
      infer_kernel<true, false, true><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
      break;
    case ((1u << 2) + (1u << 1) + 0u):
      infer_kernel<true, true, true><<<blocks, threads_per_block, shared_memory_per_block>>>(forest, out, in);
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

