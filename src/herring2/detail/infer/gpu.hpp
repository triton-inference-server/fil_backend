#pragma once
#include <kayak/buffer.hpp>
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
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  kayak::raw_index_t num_class,
  kayak::raw_index_t leaf_size
);

// See herring2/detail/infer/specializations/gpu_predict.cuh for actual
// implementation. We split this up in order to isolate CUDA code to separate
// translation units, meaning that this header can be inlcuded in host code
// units.


}
}
}
