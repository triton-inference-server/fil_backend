#pragma once
#include <cstddef>
#include <optional>
#include <herring3/postprocessor.hpp>
#include <kayak/cuda_stream.hpp>
namespace herring {

template<typename forest_t>
void predict(
  forest_t const& forest,
  postprocessor<typename forest_t::leaf_output_type, typename forest_t::io_type> const&,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  std::optional<std::size_t> specified_rows_per_block_iter=std::nullopt,
  int device=0,
  kayak::cuda_stream=kayak::cuda_stream{}
);

void initialize_gpu_options(int device=0);

}
