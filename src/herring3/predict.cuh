#pragma once
#include <cstddef>
#include <iostream>
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
  auto constexpr const preferred_tpb = size_t{256};
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
    MAX_RESIDENT_THREADS_PER_SM / threads_per_block
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

  auto resident_blocks = min(
    ceildiv(max_shared_mem_per_sm, shared_mem_per_block),
    max_resident_blocks
  );

  if (!specified_rows_per_block_iter.has_value()) {
    // Performance of this algorithm is highly sensitive to the value selected
    // for rows_per_block_iteration. This corresponds to the number of rows
    // that a single block processes before loading a new chunk of rows (if
    // necessary). wphicks was not able to find a universal formula to select
    // the optimal value for this parameter, so the following heuristics were
    // developed instead. Note that this computation is theoretically-motivated
    // but is *not* a complete theoretically-derived and empirically validated
    // solution. Future work should attempt to improve on this, and in the
    // meantime we allow users to specify a value in order to manually tune
    // performance when necessary.
    auto min_cycles = std::numeric_limits<size_t>::max();
    auto rows = rows_per_block_iteration;
    while(resident_blocks >= 2) {
      rows = rows * 2;
      auto smem = (
        row_size_bytes * rows + output_item_bytes * compute_output_size(
          row_output_size, threads_per_block, rows
        )
      );
      if (smem > max_shared_mem_per_block) {
        break;
      }

      auto blocks = ceildiv(row_count, rows);
      resident_blocks = min(
        max_shared_mem_per_sm / smem,
        max_resident_blocks
      );

      if (blocks <= resident_blocks) {
        rows_per_block_iteration = rows;
        break;
      }

      // If we're transferring less than a megabyte in total, consider compute
      // time, otherwise just go for maximum rows we can fit without dipping
      // below 2 resident blocks or exceeding shared memory limit
      if (row_size_bytes * rows * ceildiv(row_count, rows) < 1e6) {
        // Compute approximately how many cycles it will take for a block to
        // perform all of its tree inference for a single iteration. Divide by
        // the number of resident blocks to account for concurrency.
        auto tasks_per_block = forest.tree_count() * rows;
        auto compute_cycles_per_block = ceildiv(
          tasks_per_block,
          threads_per_block
        );
        auto effective_compute_cycles = ceildiv(
          compute_cycles_per_block,
          resident_blocks
        ) * blocks;

        if (effective_compute_cycles < min_cycles) {
          rows_per_block_iteration = rows;
          min_cycles = effective_compute_cycles;
        }
      } else {
        if (resident_blocks >= 2) {
          rows_per_block_iteration = rows;
        }
      }
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

  auto num_blocks = ceildiv(row_count, rows_per_block_iteration);

  /* std::cout << num_blocks << ", " << threads_per_block << ", " <<
    shared_mem_per_block << ", " << rows_per_block_iteration << ", " <<
    output_workspace_size << "\n"; */

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

