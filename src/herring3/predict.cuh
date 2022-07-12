#pragma once
#include <cstddef>
#include <iostream>
#include <stddef.h>
#include <kayak/cuda_stream.hpp>
#include <kayak/padding.hpp>
#include <herring3/exceptions.hpp>
#include <herring3/forest.hpp>
#include <herring3/gpu_introspection.hpp>
#include <herring3/shared_memory_buffer.cuh>

namespace herring {

template<
  typename leaf_output_t,
  typename node_t,
  typename io_t
>
__device__ auto evaluate_tree(
    node_t* node,
    io_t* row
) {
  while (!node->is_leaf()) {
    auto input_val = row[node->feature_index()];
    auto condition = node->default_distant();
    if (!isnan(input_val)) {
      condition = (input_val < node->threshold());
    }
    node += node->child_offset(condition);
  }
  return node->template output<leaf_output_t>();
}

template<typename forest_t>
__global__ void infer(
    forest_t forest,
    typename forest_t::io_type* output,
    typename forest_t::io_type const* input,
    size_t row_count,
    size_t col_count,
    size_t num_class,
    size_t rows_per_block_iteration,
    size_t shared_mem_byte_size,
    size_t output_workspace_size
) {
  extern __shared__ std::byte shared_mem_raw[];

  auto shared_mem = shared_memory_buffer(shared_mem_raw, shared_mem_byte_size);

  using node_t = typename forest_t::node_type;

  // TODO(wphicks): Handle vector leaves
  using output_t = std::conditional_t<
    std::is_same_v<
      typename node_t::threshold_type,
      typename forest_t::leaf_output_type
    >,
    typename node_t::threshold_type,
    typename node_t::index_type
  >;

  for (
    auto i=blockIdx.x * rows_per_block_iteration;
    i < row_count;
    i += rows_per_block_iteration * gridDim.x
  ) {

    shared_mem.clear();
    auto* output_workspace = shared_mem.fill<output_t>(output_workspace_size);

    // Handle as many rows as requested per loop or as many rows as are left to
    // process
    auto rows_in_this_iteration = min(rows_per_block_iteration, row_count - i);

    auto* input_data = shared_mem.copy(
      input + i * col_count,
      rows_in_this_iteration,
      col_count
    );

    auto trees_per_row_in_this_iteration = min(
      size_t{blockDim.x}, rows_in_this_iteration
    ) / rows_in_this_iteration;

    auto num_grove = blockDim.x / rows_in_this_iteration + (
      blockDim.x % rows_in_this_iteration != 0
    );

    // Note that this sync is safe because every thread in the block will agree
    // on whether or not a sync is required
    shared_mem.sync();

    // Infer on each tree and row
    for (
      auto task_index = threadIdx.x;
      task_index < rows_in_this_iteration * forest.tree_count();
      task_index += blockDim.x
    ) {
      auto tree_index = task_index / rows_in_this_iteration;
      auto row_index = task_index % rows_in_this_iteration;
      auto grove_index = threadIdx.x / rows_in_this_iteration;

      auto output_offset = (
        row_index * num_class * num_grove
        + (tree_index % num_class) * num_grove
        + grove_index
      );

      output_workspace[output_offset] += evaluate_tree<
        typename forest_t::leaf_output_type
      >(
        forest.get_tree_root(tree_index),
        input_data + row_index * col_count
      );
    }
  }
}

template<typename forest_t>
void predict(
  forest_t const& forest,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  int device=0,
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {
  // TODO(wphicks): Consider padding shared memory row size to odd value

  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // For Kepler or greater, this allows us to access more than 48kb of shared
  // mem per block
  // TODO(wphicks): Do this outside predict function
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer<forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );

  auto row_size_bytes = sizeof(typename forest_t::io_type) * col_count;
  auto row_output_size = max(forest.leaf_size(), class_count);
  auto single_row_output_size_bytes = sizeof(
    typename forest_t::leaf_output_type
  ) * row_output_size;

  auto threads_per_block = min(
    size_t{256},
    kayak::downpadded_size(
      (max_shared_mem_per_block  - row_size_bytes) / single_row_output_size_bytes,
      size_t{32}
    )
  );

  // If we cannot do at least a warp per block when storing input rows in
  // shared mem, recalculate our threads per block without input storage
  if (threads_per_block < 32) {
    row_size_bytes = size_t{};  // Do not store input rows in shared mem
    threads_per_block = min(
      size_t{256},
      kayak::downpadded_size(
        max_shared_mem_per_block / single_row_output_size_bytes,
        size_t{32}
      )
    );
  }

  // If we still cannot use at least a warp per block, give up
  if (threads_per_block < 32) {
    throw unusable_model_exception(
      "Model output size exceeds available shared memory"
    );
  }

  // No need to have more blocks than rows, since work on a row is never split
  // across more than one block
  auto num_blocks = min(size_t{2048}, row_count);

  // The fewest rows we can do for each loop within a block is 1, and the max
  // is the total number of rows assigned to that block
  auto min_rows_per_block_iteration = size_t{1};
  auto max_rows_per_block_iteration = row_count / num_blocks;

  // The smallest shared memory buffer we want to deal with is the size at
  // which each SM can schedule at least one block. If we were to go smaller
  // than that, it would be better to use the extra shared memory to cache more
  // data (despite the slightly less efficient block scheduling).
  auto min_smem_size = max_shared_mem_per_block / 2; // get_sm_count(device);
  auto max_smem_size = max_shared_mem_per_block;

  auto rows_per_block_iteration = min_rows_per_block_iteration;
  auto shared_mem_per_block = min_smem_size;
  auto output_workspace_size = min_smem_size;
  for (
    rows_per_block_iteration = min_rows_per_block_iteration; 
    rows_per_block_iteration <= max_rows_per_block_iteration;
    ++rows_per_block_iteration
  ) {
    output_workspace_size = (
      (single_row_output_size_bytes + rows_per_block_iteration - size_t{1}) /
      rows_per_block_iteration
    ) * rows_per_block_iteration;
    shared_mem_per_block = (
      rows_per_block_iteration * row_size_bytes + output_workspace_size
    );

    std::cout << "RPBI: " << rows_per_block_iteration << " -> " <<
      shared_mem_per_block << "\n";
    if (shared_mem_per_block >= min_smem_size) {
      break;
    }
  }

  if (shared_mem_per_block > max_smem_size) {
    --rows_per_block_iteration;
    if (rows_per_block_iteration < size_t{1}) {
      throw unusable_model_exception(
        "Model output size exceeds available shared memory"
      );
    }
    output_workspace_size = (
      (single_row_output_size_bytes + rows_per_block_iteration - size_t{1}) /
      rows_per_block_iteration
    ) * rows_per_block_iteration;
    shared_mem_per_block = (
      rows_per_block_iteration * row_size_bytes + output_workspace_size
    );
  }

  std::cout << num_blocks << ", " << threads_per_block << ", " << shared_mem_per_block << ", " << rows_per_block_iteration << "\n";

  infer<<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(
    forest,
    output,
    input,
    row_count,
    col_count,
    class_count,
    rows_per_block_iteration,
    shared_mem_per_block,
    output_workspace_size
  );
}

extern template void predict<
  forest<
    kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint16_t, float
  >
>(
  forest<
    kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint16_t, float
  > const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  int device,
  kayak::cuda_stream stream
);

}

