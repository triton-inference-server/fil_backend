#include <cstddef>
#include <optional>
#include <herring3/detail/forest.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>

namespace herring {
namespace detail {
namespace inference {

template<kayak::device_type D, typename forest_t>
std::enable_if_t<D==kayak::device_type::gpu, void> infer(
  forest_t const& forest,
  postprocessor<
    typename forest_t::leaf_output_type, typename forest_t::io_type
  > const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  std::optional<std::size_t> specified_chunk_size=std::nullopt,
  kayak::device_id<D> device=kayak::device_id<D>{},
  kayak::cuda_stream stream=kayak::cuda_stream{}
);

}
}
}
