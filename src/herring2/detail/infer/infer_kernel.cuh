#pragma once
#include <stddef.h>
#include <herring/output_ops.hpp>
#include <herring2/detail/gpu_constants.hpp>
#include <herring2/detail/postprocess.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/detail/universal_cmp.hpp>
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
  auto const num_chunks = __double2uint_ru(in.rows() * INV_CHUNK_SIZE);

  auto const warp_remainder = threadIdx.x % WARP_SIZE;
  auto const grove_index = __double2uint_rd(threadIdx.x * INV_WARP_SIZE);
  auto const num_warps = __double2uint_ru(blockDim.x * INV_WARP_SIZE);
  auto const warp_index = int(threadIdx.x * INV_WARP_SIZE);

  auto const smem_entries = shared_memory_bytes / sizeof(io_t{});

  extern __shared__ io_t workspace_mem[];
  // Zero-initialize workspace
  for (
    auto i = threadIdx.x;
    i < smem_entries;
    i += blockDim.x
  ) {
    workspace_mem[i] = io_t{};
  }

  __syncthreads();

  auto rows_per_block = uint32_t(smem_entries / (num_class * num_warps));

  auto workspace = kayak::ndarray<io_t, 2, 0, 1>(
    static_cast<io_t*>(workspace_mem), rows_per_block, num_class, num_warps
  );
  auto chunk_loop_index = 0u;
  // WH: This implies that each block MUST be able to take care of a warp's
  // worth of chunks or the total number of chunks, whichever is less
  for (
    auto chunk_index = warp_remainder + WARP_SIZE * blockIdx.x;  // WH: CHECK
    chunk_index < num_chunks;  // WH: CHECK
    chunk_index += WARP_SIZE * gridDim.x // WH: CHECK
  ) {
    auto row_start = chunk_index * CHUNK_SIZE;
    auto row_end = (chunk_index + 1) * CHUNK_SIZE;
    row_end = row_end > in.rows() ? in.rows() : row_end;
    for (
      auto tree_index = warp_index;  // WH: CHECK
      tree_index < forest.tree_count();  // WH: CHECK
      tree_index += num_warps  // WH: CHECK
    ) {
      for (auto row_index = row_start; row_index < row_end; ++row_index) {  // WH: CHECK
        auto tree_out = forest.template evaluate_tree<categorical, false, lookup>(
          tree_index,
          row_index,
          in
        );
        auto output_index = (  // WH: CHECK
          (warp_remainder + chunk_loop_index * WARP_SIZE) * CHUNK_SIZE
          + row_index - row_start
        );
        // printf("%u, %u\n", output_index, grove_index);
        /* if (blockIdx.x == 1) {
          printf("%d, %d: Storing %u to %u\n", blockIdx.x, threadIdx.x, row_index, output_index);
        } */
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
          /* printf("%d, %d, %d, %u, %u, %u, %u, %u, %f\n", blockIdx.x,
              warp_index, threadIdx.x, chunk_index, row_index, tree_index,
              output_index, grove_index,
              float(tree_out.at(0))); */
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
      warp_remainder
      + blockIdx.x * WARP_SIZE
      + warp_index * WARP_SIZE * gridDim.x
    ); // WH: CHECK
    chunk_index < num_chunks; // WH: CHECK
    chunk_index += blockDim.x * gridDim.x * num_warps // WH: CHECK
  ) {
    auto workspace_chunk_index = (
      threadIdx.x + loop_index * blockDim.x
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

      // printf("%d: Retrieving row %u from %u\n", blockIdx.x, row_index, workspace_row_index);
      for (auto i = raw_index_t{}; i < final_output.size(); ++i) {
        out.at(row_index, i) = final_output.at(i);
        // printf("%u: %f\n", row_index, final_output.at(i));
      }
    }

    break;
    loop_index += 1;
  }
}

}
}
}
}
