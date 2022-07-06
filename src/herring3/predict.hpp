#pragma once
#include <kayak/cuda_stream.hpp>
namespace herring {

template<typename forest_t, typename io_t>
void predict(
  forest_t const& forest,
  io_t* output,
  io_t* input,
  typename forest_t::index_type row_count,
  typename forest_t::index_type col_count,
  typename forest_t::index_type class_count,
  int device=0,
  kayak::cuda_stream=kayak::cuda_stream{}
);

}
