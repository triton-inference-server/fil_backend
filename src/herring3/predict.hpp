#pragma once
#include <cstddef>
#include <kayak/cuda_stream.hpp>
namespace herring {

template<typename forest_t>
void predict(
  forest_t const& forest,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  int device=0,
  kayak::cuda_stream=kayak::cuda_stream{}
);

}
