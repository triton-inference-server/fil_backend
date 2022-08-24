#include <cstddef>
#include <herring3/constants.hpp>
#include <herring3/detail/forest.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <herring3/detail/infer/gpu.cuh>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <kayak/tree_layout.hpp>
namespace herring {
namespace detail {
namespace inference {

template void infer<
  kayak::device_type::gpu,
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >
>(
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  > const&,
  postprocessor<float> const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  std::nullptr_t,
  std::optional<std::size_t>,
  kayak::device_id<kayak::device_type::gpu>,
  kayak::cuda_stream stream
);

}
}
}
