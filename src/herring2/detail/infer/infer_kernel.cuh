#pragma once
#include <herring2/detail/gpu_constants.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>

namespace herring {
namespace detail {
namespace inference {

using kayak::raw_index_t;

template <
  bool categorical,
  bool lookup,
  bool vector_leaf,
  typename forest_t,
  typename input_t,
  typename output_t
>
__global__ void infer_kernel(forest_t forest,
    kayak::data_array<kayak::data_layout::dense_row_major, output_t> out,
    kayak::data_array<kayak::data_layout::dense_row_major, input_t> in,
    raw_index_t num_class
) {
  auto task_id = blockIdx.x * blockDim.x + threadIdx.x;

  auto const num_chunks = __double2uint_ru(in.rows() * INV_CHUNK_SIZE);

  auto const warp_remainder = task_id % WARP_SIZE;

  extern __shared__ typename forest_t::output_type workspace_mem[];

  auto rows_per_block = padded_size(
    padded_size(in.rows(), CHUNK_SIZE) / CHUNK_SIZE, gridDim.x
  ) / gridDim.x;

  auto workspace = kayak::data_array<kayak::data_layout::dense_row_major, typename forest_t::output_t>(
    workspace_mem, rows_per_block, num_class
  );
  auto loop_index = 0u;
  for (
    auto chunk_index = warp_remainder + WARP_SIZE * blockIdx.x;
    chunk_index < num_chunks;
    chunk_index += WARP_SIZE * gridDim.x;
  ) {
    auto row_start = chunk_index * CHUNK_SIZE;
    auto row_end = (chunk_index + 1) * CHUNK_SIZE;
    row_end = row_end > in.rows() ? in.rows() : row_end;
    for (
      auto tree_index = int{task_id * INV_WARP_SIZE};
      tree_index < forest.tree_count();
      tree_index += WARP_SIZE * blockDim.x;
    ) {
      for (auto row = row_start; row < row_end; ++row) {
        auto tree_out = forest.evaluate_tree<categorical, false, lookup>(
          tree_index,
          row_index,
          input
        );
        auto output_index = (
          (loop_index * WARP_SIZE + warp_remainder) * CHUNK_SIZE
          + row - row_start
        );
        if constexpr (vector_leaf) {
          for (auto class_index = 0u; class_index < num_class; ++class_index) {
            workspace.at(output_index, class_index) = tree_out.at(class_index);
          }
        } else {
          auto class_index = tree_index % num_class;
          workspace.at(output_index, class_index) = tree_out.at(0);
        }
      }
    }
    ++loop_index;
  }

  __syncthreads();
  /* After sync, we know that all the rows processed in this block have been
   * fully processed by all trees (since every block processes all trees). We
   * can now aggregate the results from each tree back into the provided output
   * memory for rows assigned to this block. */

  loop_index = 0u;
  for (
    auto chunk_index = (
      warp_remainder + int{task_id * INV_WARP_SIZE} * WARP_SIZE * gridDim.x
    );
    chunk_index < num_chunks;
    chunk_index += blockDim.x * gridDim.x;
  ) {
    auto row_start = chunk_index * CHUNK_SIZE;
    auto row_end = (chunk_index + 1) * CHUNK_SIZE;
    row_end = row_end > in.rows() ? in.rows() : row_end;

    // TODO(wphicks): Aggregate results from each tree for each row from
    // workspace into out
    ++loop_index;
  }
}

}
}
}
