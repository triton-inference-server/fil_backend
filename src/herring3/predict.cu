#include <cstddef>
#include <herring3/postprocessor.hpp>
#include <herring3/predict.cuh>
#include <kayak/cuda_stream.hpp>
#include <kayak/tree_layout.hpp>
namespace herring {

template void predict<
  forest<
    kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint16_t, float
  >
>(
  forest<
    kayak::tree_layout::depth_first, float, uint32_t, uint16_t, uint16_t, float
  > const&,
  postprocessor<float, float> const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  std::size_t,
  int device,
  kayak::cuda_stream stream
);

}
