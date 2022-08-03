#pragma once
#include <cstddef>
#include <iostream>
#include <numeric>
#include <optional>
#include <stddef.h>
#include <kayak/cuda_stream.hpp>
#include <kayak/padding.hpp>
#include <herring3/ceildiv.hpp>
#include <herring3/exceptions.hpp>
#include <herring3/forest.hpp>
#include <herring3/gpu_introspection.hpp>
#include <herring3/postprocessor.hpp>
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
  auto cur_node = *node;
  while (!cur_node.is_leaf()) {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      condition = (input_val < cur_node.threshold());
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  }
  return cur_node.template output<leaf_output_t>();
}

template<typename forest_t>
__global__ void infer(
    forest_t forest,
    postprocessor<
      typename forest_t::leaf_output_type, typename forest_t::io_type
    > postproc,
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

  using io_t = typename forest_t::io_type;

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

    auto task_count = rows_in_this_iteration * forest.tree_count();

    auto num_grove = ceildiv(
      min(size_t{blockDim.x}, task_count),
      rows_in_this_iteration
    );

    // Note that this sync is safe because every thread in the block will agree
    // on whether or not a sync is required
    shared_mem.sync();

    // Every thread must iterate the same number of times in order to avoid a
    // deadlock on __syncthreads, so we round the task_count up to the next
    // multiple of the number of threads in this block. We then only perform
    // work within the loop if the task_index is below the actual task_count.
    auto const task_count_rounded_up = blockDim.x * ceildiv(task_count, blockDim.x);

    // Infer on each tree and row
    for (
      auto task_index = threadIdx.x;
      task_index < task_count_rounded_up;
      task_index += blockDim.x
    ) {
      auto row_index = task_index % rows_in_this_iteration;
      auto tree_index = task_index / rows_in_this_iteration;
      auto grove_index = threadIdx.x / rows_in_this_iteration;

      auto output_offset = (
        row_index * num_class * num_grove
        + (tree_index % num_class) * num_grove
        + grove_index
      ) * (task_index < task_count);

      output_workspace[output_offset] += evaluate_tree<
        typename forest_t::leaf_output_type
      >(
        forest.get_tree_root(tree_index),
        input_data + row_index * col_count
      ) * (task_index < task_count);
      __syncthreads();
    }

    task_count = rows_in_this_iteration * num_class;

    for (
      auto task_index = threadIdx.x;
      task_index < task_count;
      task_index += blockDim.x
    ) {
      auto row_index = task_index % rows_in_this_iteration;
      auto class_index = task_index / rows_in_this_iteration;
      auto grove_offset = (
        row_index * num_class * num_grove + class_index * num_grove
      );
      for (
        auto grove_index = size_t{1};
        grove_index < num_grove;
        ++grove_index
      ) {
        output_workspace[grove_offset] += output_workspace[
          grove_offset + grove_index
        ];
      }
    }
    __syncthreads();
    for (
      auto row_index = threadIdx.x;
      row_index < rows_in_this_iteration;
      row_index += blockDim.x
    ) {
      postproc(
        output_workspace + (row_index) * num_class * num_grove,
        num_class, 
        output + ((i + row_index) * num_class)
      );
    }
    __syncthreads();
  }
}

auto compute_output_size(
  size_t row_output_size,
  size_t threads_per_block,
  size_t rows_per_block_iteration
) {
  return row_output_size * ceildiv(
    threads_per_block,
    rows_per_block_iteration
  ) * rows_per_block_iteration;
}

template<typename forest_t>
void predict(
  forest_t const& forest,
  postprocessor<
    typename forest_t::leaf_output_type, typename forest_t::io_type
  > const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  std::optional<std::size_t> specified_rows_per_block_iter=std::nullopt,
  int device=0,
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {

  auto sm_count = get_sm_count(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  auto max_shared_mem_per_sm = get_max_shared_mem_per_sm(device);
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
  auto row_output_size_bytes = sizeof(
    typename forest_t::leaf_output_type
  ) * row_output_size;

  // First determine the number of threads per block. This is the indicated
  // preferred value unless we cannot handle at least 1 row per block iteration
  // with available shared memory, in which case must reduce the threads per
  // block.
  auto constexpr const preferred_tpb = size_t{512};
  auto threads_per_block = min(
    preferred_tpb,
    kayak::downpadded_size(
      (max_shared_mem_per_block  - row_size_bytes) / row_output_size_bytes,
      size_t{32}
    )
  );

  // If we cannot do at least a warp per block when storing input rows in
  // shared mem, recalculate our threads per block without input storage
  if (threads_per_block < 32) {
    std::cout << "Not enough room for input data in smem\n";
    row_size_bytes = size_t{};  // Do not store input rows in shared mem
    threads_per_block = min(
      preferred_tpb,
      kayak::downpadded_size(
        max_shared_mem_per_block / row_output_size_bytes,
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

  auto const max_resident_blocks = sm_count * (
    get_max_threads_per_sm(device) / threads_per_block
  );

  // Compute shared memory usage based on minimum or specified
  // rows_per_block_iteration
  auto rows_per_block_iteration = specified_rows_per_block_iter.value_or(
    size_t{1}
  );
  auto constexpr const output_item_bytes = sizeof(
    typename forest_t::leaf_output_type
  );
  auto output_workspace_size = compute_output_size(
    row_output_size, threads_per_block, rows_per_block_iteration
  );
  auto output_workspace_size_bytes = output_item_bytes * output_workspace_size;
  if (output_workspace_size_bytes > max_shared_mem_per_block) {
    throw unusable_model_exception(
      "Model output size exceeds available shared memory"
    );
  }
  auto shared_mem_per_block = min(
    rows_per_block_iteration * row_size_bytes + output_workspace_size_bytes,
    max_shared_mem_per_block
  );

  auto resident_blocks_per_sm = min(
    ceildiv(max_shared_mem_per_sm, shared_mem_per_block),
    max_resident_blocks
  );

  // If caller has not specified the number of rows per block iteration, apply
  // the following heuristic to identify an approximately optimal value
  if (
    !specified_rows_per_block_iter.has_value()
    && resident_blocks_per_sm >= 2
  ) {
    auto prev_align_penalty = row_count * row_size_bytes;
    // Iterate through powers of two for rows_per_block_iteration up to the
    // value that guarantees aligned chunks for every block iteration
    for (
      auto rpbi=size_t{2};
      rpbi < MAX_READ_CHUNK / sizeof(typename forest_t::io_type);
      rpbi <<= 1
    ) {
      auto smem = output_item_bytes * compute_output_size(
        row_output_size, threads_per_block, rows_per_block_iteration
      );
      if (smem > max_shared_mem_per_block) {
        break;
      }
      auto res_blocks = min(
        ceildiv(max_shared_mem_per_sm, smem),
        max_resident_blocks
      );
      if (res_blocks < 2) {
        break;
      }

      auto total_block_iters = ceildiv(row_count, rpbi);
      // Determine approximately how often block iterations will start on an
      // unaligned read
      auto align_overlap = (row_size_bytes * rpbi) % MAX_READ_CHUNK;
      auto align_load_interval = size_t{row_size_bytes != 0} + std::lcm(
        align_overlap, MAX_READ_CHUNK
      ) / MAX_READ_CHUNK;
      auto total_loads = ceildiv(
        total_block_iters * row_size_bytes,
        MAX_READ_CHUNK
      );
      auto align_penalty = total_loads - total_loads / align_load_interval;
      // Only worth increasing rows if we have a significant decrease in
      // alignment penalty
      if (prev_align_penalty < 2 * col_count + align_penalty) {
        break;
      }
      rows_per_block_iteration = rpbi;
      if (res_blocks > total_block_iters) {
        break;
      }
      prev_align_penalty = align_penalty;
    }
  }

  output_workspace_size = compute_output_size(
    row_output_size, threads_per_block, rows_per_block_iteration
  );
  output_workspace_size_bytes = output_item_bytes * output_workspace_size;

  shared_mem_per_block = (
    rows_per_block_iteration * row_size_bytes + output_workspace_size_bytes
  );

  // Divide shared mem evenly
  shared_mem_per_block = max_shared_mem_per_block / (
    max_shared_mem_per_block / shared_mem_per_block
  );

  auto num_blocks = std::min(
    ceildiv(row_count, rows_per_block_iteration),
    MAX_BLOCKS
  );

  std::cout << rows_per_block_iteration << "\n";

  infer<<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(
    forest,
    postproc,
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
  postprocessor<float, float> const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  std::optional<std::size_t>,
  int device,
  kayak::cuda_stream stream
);

}

