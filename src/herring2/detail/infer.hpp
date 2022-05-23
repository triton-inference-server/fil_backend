#pragma once
#include <herring2/detail/infer/cpu.hpp>
#ifdef ENABLE_GPU
#include <kayak/detail/infer/gpu.hpp>
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
  raw_index_t leaf_size
) {
  inference::predict<D>(forest, out, in, num_class, leaf_size);
}

}
}
