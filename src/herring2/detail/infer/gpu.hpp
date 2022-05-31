#pragma once
#include <herring/output_ops.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <type_traits>

namespace herring {
namespace detail {
namespace inference {

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
>
std::enable_if_t<D == kayak::device_type::gpu && kayak::GPU_ENABLED, void> predict(
  forest_t const& model_forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  kayak::raw_index_t num_class,
  element_op element_postproc,
  row_op row_postproc,
  io_t average_factor,
  io_t bias,
  io_t postproc_constant,
  int device_id,
  kayak::cuda_stream stream
);

// See herring2/detail/infer/gpu_predict.cuh for actual implementation. We
// split this up in order to isolate CUDA code to separate translation units,
// meaning that this header can be included in host code units.


}
}
}
