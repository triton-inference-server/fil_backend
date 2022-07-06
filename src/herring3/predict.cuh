#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <iostream>
#include <herring3/exceptions.hpp>
#include <herring3/forest.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/padding.hpp>
#include <kayak/tree_layout.hpp>

namespace herring {

using raw_index_t = kayak::raw_index_t;
using byte = char;

inline auto get_max_shared_mem_per_block(int device_id) {
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

inline auto get_sm_count(int device_id) {
  auto result = int{};
  kayak::cuda_check(
    cudaDeviceGetAttribute(
      &result,
      cudaDevAttrMultiProcessorCount,
      device_id
    )
  );
  return raw_index_t(result);
}

/* Copy the given number of rows into shared memory iff they will fit. If so,
 * input_data is updated to point to the shared memory location,
 * shared_mem_dest is updated to point to the memory location past the end of
 * the copied memory, and shared_mem_size_bytes is updated to indicate the
 * remaining amount of shared memory.
 */
template<typename io_t>
__device__ auto copy_rows_to_shared_memory(
  io_t* input_data,
  byte* shared_mem_dest,
  size_t shared_mem_size_bytes,
  raw_index_t rows_to_copy,
  raw_index_t col_count
) {
  auto* smem = reinterpret_cast<io_t*>(shared_mem_dest);
  auto entries_to_copy = rows_to_copy * col_count;
  entries_to_copy *= (entries_to_copy * sizeof(io_t) <= shared_mem_size_bytes);
  for(auto i = threadIdx.x; i < entries_to_copy; i += blockDim.x) {
    smem[i] = input_data[i];
  }
  __syncthreads();
  return entries_to_copy;
}

template<typename T>
__device__ auto fill_buffer(
  T* buffer,
  size_t entries,
  T value = T{}
) {
  for(auto i = threadIdx.x; i < entries; i += blockDim.x) {
    buffer[i] = value;
  }
}

template<typename T>
__device__ auto copy_data_to_shared_memory(
  T* data,
  byte* shared_mem_dest,
  size_t shared_mem_size_bytes,
  raw_index_t entries_to_copy
) {
  auto* smem = reinterpret_cast<T*>(shared_mem_dest);
  entries_to_copy *= (entries_to_copy * sizeof(T) <= shared_mem_size_bytes);
  for(auto i = threadIdx.x; i < entries_to_copy; i += blockDim.x) {
    smem[i] = data[i];
  }
  __syncthreads();
  return entries_to_copy;
}

template<
  typename node_value_t,
  typename feature_index_t,
  typename offset_t,
  typename io_t,
  typename output_t
>
__device__ void evaluate_tree(
    node_value_t* node_value_p,
    feature_index_t* node_feature_p,
    offset_t* distant_child_offset_p,
    node_metadata_storage* metadata_p,
    io_t* input,
    output_t* output,
    raw_index_t num_class,
    raw_index_t output_size,
    raw_index_t output_index=raw_index_t{},
    output_t* output_leaves_p=nullptr
) {
  auto distant_offset = *distant_child_offset_p;

  while (!metadata_p->is_leaf()) {
    auto condition = false;
    auto feature_value = input[*node_feature_p];
    if (isnan(feature_value)) {
      condition = metadata_p->default_distant();
    } else {
      condition = (feature_value < node_value_p->value);
    }

    auto offset_to_next_node = 1u + condition * (distant_offset - 1u);
    node_value_p += offset_to_next_node;
    node_feature_p += offset_to_next_node;
    distant_child_offset_p += offset_to_next_node;
    metadata_p += offset_to_next_node;

    distant_offset = *distant_child_offset_p;
  }
  if constexpr (std::is_same_v<output_t, typename node_value_t::value_type>) {
    if (output_size == 1) {
      *output += node_value_p->value;
    } else {
      for (auto i=output_index; i < output_index + output_size; ++i) {
        output[i] += output_leaves_p[node_value_p->index + output_index];
      }
    }
  } else if constexpr (std::is_same_v<output_t, typename node_value_t::output_index_type>) {
    if (output_size == 1) {
      *output += node_value_p->index;
    } else {
      for (auto i=output_index; i < output_index + output_size; ++i) {
        output[i] += output_leaves_p[node_value_p->index + output_index];
      }
    }
  } else {
    for (auto i=output_index; i < output_index + output_size; ++i) {
      output[i] += output_leaves_p[node_value_p->index + output_index];
    }
  }
}

template<typename forest_t, typename io_t>
__global__ void infer(
    forest_t forest,
    io_t* output,
    io_t* input,
    raw_index_t row_count,
    raw_index_t col_count,
    raw_index_t num_class,
    raw_index_t rows_per_block_iteration,
    size_t shared_mem_byte_size,
    size_t output_workspace_size
) {
  extern __shared__ byte shared_mem_buffer[];
  // The first part of the shared memory buffer is reserved for storing the
  // output of each tree in an iteration
  auto* output_workspace = reinterpret_cast<typename forest_t::output_type*>(shared_mem_buffer);

  // The rest of the shared memory buffer can be used to cache
  // frequently-accessed values in each iteration
  auto* shared_mem_cache = (
    reinterpret_cast<byte*>(output_workspace + output_workspace_size)
  );
  shared_mem_byte_size -= output_workspace_size * sizeof(
    typename forest_t::output_type
  );

  for (
    auto i=blockIdx.x * rows_per_block_iteration;
    i < row_count;
    i += rows_per_block_iteration * gridDim.x
  ) {

    // Clear results from previous rows
    fill_buffer(output_workspace, output_workspace_size);
    // Reset the shared mem cache at the beginning of each iteration
    auto* shared_mem_remainder = shared_mem_cache;
    auto shared_mem_remainder_size = shared_mem_byte_size;

    // Handle as many rows as requested per loop or as many rows as are left to
    // process
    auto rows_in_this_iteration = raw_index_t(max(
      0, min(int{rows_per_block_iteration}, int{row_count} - int{i})
    ));

    auto* input_data = input + i * col_count;

    // Attempt to copy input data to shared memory
    auto entries_copied = copy_rows_to_shared_memory(
      input_data,
      shared_mem_remainder,
      shared_mem_remainder_size,
      rows_in_this_iteration,
      col_count
    );
    input_data = (entries_copied == 0) ? input_data : reinterpret_cast<io_t*>(shared_mem_remainder);
    shared_mem_remainder += entries_copied * sizeof(io_t);
    shared_mem_remainder_size = shared_mem_byte_size - entries_copied * sizeof(io_t);

    auto threads_per_tree = min(blockDim.x, rows_in_this_iteration);

    // The number of trees this block will handle with each loop over tasks
    auto trees_per_iteration = blockDim.x / threads_per_tree;

    // Infer on each tree and row
    for (
      auto task_index = threadIdx.x;
      task_index < rows_in_this_iteration * forest.tree_count_;
      task_index += blockDim.x
    ) {
      auto tree_index = task_index / threads_per_tree;
      auto row_index = task_index % rows_in_this_iteration;

      auto output_offset = (
        (row_index % rows_in_this_iteration) * trees_per_iteration * num_class
        + (tree_index % trees_per_iteration) * num_class
      );

      auto tree_offset = forest.tree_offsets_[tree_index];
      evaluate_tree(
        forest.values_ + tree_offset,
        forest.features_ + tree_offset,
        forest.distant_offsets_ + tree_offset,
        forest.metadata_ + tree_offset,
        input_data + row_index * col_count,
        output_workspace + output_offset,
        num_class,
        forest.output_size_,
        (tree_index % num_class) * (forest.output_size_ == 1),
        forest.outputs_
      );
    }
  }
}

template<typename forest_t, typename io_t>
void predict(
  forest_t const& forest,
  io_t* output,
  io_t* input,
  typename forest_t::index_type row_count,
  typename forest_t::index_type col_count,
  typename forest_t::index_type class_count,
  int device=0,
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {
  // TODO(wphicks): Consider padding shared memory row size to odd value

  // TODO(wphicks): Do this outside predict function
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // For Kepler or greater, this allows us to access more than 48kb of shared
  // mem per block
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer<forest_t, io_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );

  auto row_size_bytes = sizeof(io_t) * col_count;
  auto row_output_size = max(forest.output_size_, class_count);
  auto row_output_size_bytes = sizeof(
    typename forest_t::output_type
  ) * row_output_size;

  auto threads_per_block = min(
    256,
    kayak::downpadded_size(
      int{max_shared_mem_per_block / row_output_size_bytes},
      32
    )
  );

  if (threads_per_block < 32) {
    throw unusable_model_exception(
      "Model output size exceeds available shared memory"
    );
  }

  auto num_blocks = min(2048, int{row_count.value()});

  auto output_workspace_size = (
    row_output_size * min(threads_per_block, forest.tree_count_)
  );
  auto output_workspace_size_bytes = sizeof(
    typename forest_t::output_type
  ) * output_workspace_size;

  auto shared_mem_per_block = min(  // No more than max available
    max_shared_mem_per_block,
    max( // No less than required for one row and its output
      int{max_shared_mem_per_block / get_sm_count(device)},
      int{output_workspace_size_bytes + row_size_bytes}
    )
  );

  // auto num_blocks = 1;
  // auto threads_per_block = 1;

  // The number of rows each block processes per loop of execution. When
  // calculating the number of blocks we can handle in shared memory, we must
  // first subtract off the space required to store the output.
  //
  // Assuming we sync at the end of each iteration of trees, we need as much
  // memory as there are trees in an iteration times the output size required
  // for each row times the number of rows in each iteration. Since the number
  // of trees per iteration is the number of threads per block divided by the
  // number of rows per iteration, we are left with a footprint of the number
  // of threads per block times the output size for a single row.
  auto rows_per_block_iteration = max(
    1,
    // No need to do more blocks per iteration than there are rows per block,
    // but otherwise take care of as many rows per iteration as we have space
    // for in shared memory
    min(
      (
        int{shared_mem_per_block}
        - int{output_workspace_size_bytes}
      ) / int{row_size_bytes},
      int{row_count.value()} / num_blocks
    )
  );

  std::cout << num_blocks << ", " << threads_per_block << ", " << shared_mem_per_block << ", " << rows_per_block_iteration << "\n";

  infer<<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(
    forest,
    output,
    input,
    row_count.value(),
    col_count.value(),
    class_count.value(),
    rows_per_block_iteration,
    shared_mem_per_block,
    output_workspace_size
  );
}

extern template void predict<
  forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>,
  float
>(
  forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false> const&,
  float*,
  float*,
  kayak::detail::index_type<false>,
  kayak::detail::index_type<false>,
  kayak::detail::index_type<false>,
  int device,
  kayak::cuda_stream stream
);

}

