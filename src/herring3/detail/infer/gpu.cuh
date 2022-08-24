#pragma once
#include <cstddef>
#include <optional>
#include <herring3/detail/gpu_introspection.hpp>
#include <herring3/detail/infer_kernel/gpu.cuh>
#include <herring3/detail/postprocessor.hpp>
#include <herring3/exceptions.hpp>
#include <kayak/ceildiv.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/padding.hpp>

namespace herring {
namespace detail {
namespace inference {

inline auto compute_output_size(
  size_t row_output_size,
  size_t threads_per_block,
  size_t rows_per_block_iteration
) {
  return row_output_size * kayak::ceildiv(
    threads_per_block,
    rows_per_block_iteration
  ) * rows_per_block_iteration;
}

template<kayak::device_type D, typename forest_t, typename vector_output_t=std::nullptr_t>
std::enable_if_t<D==kayak::device_type::gpu, void> infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  vector_output_t vector_output=nullptr,
  std::optional<std::size_t> specified_chunk_size=std::nullopt,
  kayak::device_id<D> device=kayak::device_id<D>{},
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {

  // std::cout << "Trees: " << forest.tree_count() << ", Rows: " << row_count << "\n";
  auto sm_count = get_sm_count(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  auto max_shared_mem_per_sm = get_max_shared_mem_per_sm(device);

  auto row_size_bytes = sizeof(typename forest_t::io_type) * col_count;
  auto row_output_size = max(forest.leaf_size(), class_count);
  auto row_output_size_bytes = sizeof(
    typename forest_t::io_type
  ) * row_output_size;

  // First determine the number of threads per block. This is the indicated
  // preferred value unless we cannot handle at least 1 row per block iteration
  // with available shared memory, in which case we must reduce the threads per
  // block.
  auto constexpr const preferred_tpb = size_t{512};
  auto threads_per_block = min(
    preferred_tpb,
    kayak::downpadded_size(
      (max_shared_mem_per_block  - row_size_bytes) / row_output_size_bytes,
      WARP_SIZE
    )
  );

  // If we cannot do at least a warp per block when storing input rows in
  // shared mem, recalculate our threads per block without input storage
  if (threads_per_block < WARP_SIZE) {
    row_size_bytes = size_t{};  // Do not store input rows in shared mem
    threads_per_block = min(
      preferred_tpb,
      kayak::downpadded_size(
        max_shared_mem_per_block / row_output_size_bytes,
        WARP_SIZE
      )
    );
  }

  // If we still cannot use at least a warp per block, give up
  if (threads_per_block < WARP_SIZE) {
    throw unusable_model_exception(
      "Model output size exceeds available shared memory"
    );
  }

  auto const max_resident_blocks = sm_count * (
    get_max_threads_per_sm(device) / threads_per_block
  );

  // Compute shared memory usage based on minimum or specified
  // rows_per_block_iteration
  auto rows_per_block_iteration = specified_chunk_size.value_or(
    size_t{1}
  );
  auto constexpr const output_item_bytes = sizeof(
    typename forest_t::io_type
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
    kayak::ceildiv(max_shared_mem_per_sm, shared_mem_per_block),
    max_resident_blocks
  );

  // If caller has not specified the number of rows per block iteration, apply
  // the following heuristic to identify an approximately optimal value
  if (
    !specified_chunk_size.has_value()
    && resident_blocks_per_sm >= 2
  ) {
    rows_per_block_iteration = size_t{32};
  }

  output_workspace_size = compute_output_size(
    row_output_size, threads_per_block, rows_per_block_iteration
  );
  output_workspace_size_bytes = output_item_bytes * output_workspace_size;

  shared_mem_per_block = (
    rows_per_block_iteration * row_size_bytes + output_workspace_size_bytes
  );

  // Divide shared mem evenly
  shared_mem_per_block = max_shared_mem_per_sm / (
    max_shared_mem_per_sm / shared_mem_per_block
  );

  auto num_blocks = std::min(
    kayak::ceildiv(row_count, rows_per_block_iteration),
    MAX_BLOCKS
  );
  infer_kernel<false><<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(
    forest,
    postproc,
    output,
    input,
    row_count,
    col_count,
    class_count,
    shared_mem_per_block,
    output_workspace_size,
    rows_per_block_iteration,
    vector_output
  );
}

extern template void infer<
  kayak::device_type::gpu,
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >
>(
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  > const&,
  postprocessor<float> const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  float*,
  std::optional<std::size_t>,
  kayak::device_id<kayak::device_type::gpu>,
  kayak::cuda_stream stream
);

}
}
}
