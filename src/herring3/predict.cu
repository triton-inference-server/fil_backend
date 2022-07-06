#include <herring3/predict.cuh>
#include <kayak/tree_layout.hpp>
namespace herring {

template void predict<
  forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false>,
  float
>(
  forest<kayak::tree_layout::depth_first, float, uint16_t, uint16_t, uint32_t, float, false> const&,
  float*,
  float*,
  kayak::detail::index_type<false>,
  kayak::detail::index_type<false>,
  kayak::detail::index_type<false>,
  int device,
  kayak::cuda_stream stream
);

}
