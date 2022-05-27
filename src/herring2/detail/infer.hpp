#pragma once
#include <herring2/detail/infer/cpu.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/infer/gpu.hpp>
#endif
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/data_array.hpp>

namespace herring {
namespace detail {

using kayak::raw_index_t;

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
> void predict(
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  raw_index_t num_class,
  element_op element_postproc,
  row_op row_postproc,
  io_t average_factor,
  io_t bias,
  io_t postproc_constant,
  int device_id=0,
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {
  inference::predict<D>(forest, out, in, num_class, element_postproc, row_postproc, average_factor, bias, postproc_constant, device_id, stream);
}

}
}
