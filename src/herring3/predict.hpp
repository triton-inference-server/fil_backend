#pragma once
#include <cstddef>
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
  int device=0,
  kayak::cuda_stream=kayak::cuda_stream{}
);

}
