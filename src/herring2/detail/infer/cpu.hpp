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

using kayak::raw_index_t;

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
>
std::enable_if_t<D == kayak::device_type::cpu, void> predict(
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  raw_index_t num_class,
  element_op element_postproc,
  row_op row_postproc,
  io_t average_factor,
  io_t bias,
  io_t postproc_constant,
  int device_id,
  kayak::cuda_stream stream
);

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
>
std::enable_if_t<D == kayak::device_type::gpu && !kayak::GPU_ENABLED, void> predict(
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  raw_index_t num_class,
  element_op element_postproc,
  row_op row_postproc,
  io_t average_factor,
  io_t bias,
  io_t postproc_constant,
  int device_id,
  kayak::cuda_stream stream
) {
  throw kayak::gpu_unsupported(
    "Attempting to launch forest inference on device in non-GPU build"
  );
}

}
}
}
