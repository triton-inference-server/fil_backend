#pragma once
#include <stddef.h>
#include <herring/output_ops.hpp>
#include <herring2/detail/gpu_constants.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <herring2/detail/postprocess.hpp>
#include <kayak/ndarray.hpp>
#include <kayak/padding.hpp>

namespace herring {
namespace detail {
namespace inference {
namespace gpu {

using kayak::raw_index_t;

template <
  bool categorical,
  bool lookup,
  bool vector_leaf,
  typename forest_t,
  typename io_t
>
__global__ void infer_kernel(
    forest_t forest,
    kayak::data_array<kayak::data_layout::dense_row_major, io_t> out,
    kayak::data_array<kayak::data_layout::dense_row_major, io_t> in,
    raw_index_t num_class,
    element_op element_postproc,
    row_op row_postproc,
    io_t average_factor,
    io_t bias,
    io_t postproc_constant,
    size_t shared_memory_bytes
) {
  auto task_id = blockIdx.x * blockDim.x + threadIdx.x;

  auto const num_chunks = __double2uint_ru(in.rows() * INV_CHUNK_SIZE);

  auto const warp_remainder = task_id % WARP_SIZE;
  auto const grove_index = __double2uint_ru(threadIdx.x * INV_WARP_SIZE);
  auto const num_warps = __double2uint_ru(blockDim.x * INV_WARP_SIZE);
  auto const num_groves = forest.tree_count() / num_warps;

  extern __shared__ io_t workspace_mem[];
  // Zero-initialize workspace
  for (
    auto i = threadIdx.x;
    i < shared_memory_bytes / sizeof(io_t{});
    i += blockDim.x
  ) {
    workspace_mem[i] = io_t{};
  }

  __syncthreads();

  auto rows_per_block = CHUNK_SIZE * kayak::padded_size(
    kayak::padded_size(in.rows(), CHUNK_SIZE) / CHUNK_SIZE, gridDim.x
  ) / gridDim.x;

  auto workspace = kayak::ndarray<io_t, 2, 0, 1>(
    static_cast<io_t*>(workspace_mem), rows_per_block, num_class, num_groves
  );
  auto chunk_loop_index = 0u;
  for (
    auto chunk_index = warp_remainder + WARP_SIZE * blockIdx.x;
    chunk_index < num_chunks;
    chunk_index += WARP_SIZE * gridDim.x
  ) {
    auto row_start = chunk_index * CHUNK_SIZE;
    auto row_end = (chunk_index + 1) * CHUNK_SIZE;
    row_end = row_end > in.rows() ? in.rows() : row_end;
    for (
      auto tree_index = int(task_id * INV_WARP_SIZE);
      tree_index < forest.tree_count();
      tree_index += WARP_SIZE * blockDim.x
    ) {
      for (auto row_index = row_start; row_index < row_end; ++row_index) {
        auto tree_out = forest.template evaluate_tree<categorical, false, lookup>(
          tree_index,
          row_index,
          in
        );
        auto output_index = (
          chunk_loop_index * CHUNK_SIZE
          + row_index - row_start
        );
        // Every thread within the warp has a different output_index and every
        // warp has a different grove_index, so there will be no collisions for
        // incrementing outputs
        if constexpr (vector_leaf) {
          for (auto class_index = 0u; class_index < num_class; ++class_index) {
            workspace.at(output_index, class_index, grove_index) += tree_out.at(class_index);
          }
        } else {
          auto class_index = tree_index % num_class;
          workspace.at(output_index, class_index, grove_index) += tree_out.at(0);
        }
      }
    }
    ++chunk_loop_index;
  }

  __syncthreads();
  /* After sync, we know that all the rows processed in this block have been
   * fully processed by all trees (since every block processes all trees). We
   * can now aggregate the results from each tree back into the provided output
   * memory for rows assigned to this block. */

  auto loop_index = 0u;
  for (
    auto chunk_index = (
      warp_remainder + int(task_id * INV_WARP_SIZE) * WARP_SIZE * gridDim.x
    );
    chunk_index < num_chunks;
    chunk_index += blockDim.x * gridDim.x
  ) {
    auto workspace_chunk_index = (
      threadIdx.x + loop_index * blockDim.x * gridDim.x
    );
    auto row_start = chunk_index * CHUNK_SIZE;
    auto row_end = (chunk_index + 1) * CHUNK_SIZE;
    row_end = row_end > in.rows() ? in.rows() : row_end;
    for (auto row_index = row_start; row_index < row_end; ++row_index) {
      auto workspace_row_index = (
        workspace_chunk_index * CHUNK_SIZE + row_index - row_start
      );
      auto final_output = postprocess(
        workspace,
        workspace_row_index,
        element_postproc,
        row_postproc,
        average_factor,
        bias,
        postproc_constant
      );

      for (auto i = raw_index_t{}; i < final_output.size(); ++i) {
        out.at(row_index, i) = final_output.at(i);
      }
    }

    loop_index += 1;
  }
}

}
}
}
}
