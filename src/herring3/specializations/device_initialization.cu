#include <stdint.h>
#include <herring3/constants.hpp>
#include <herring3/detail/device_initialization/gpu.cuh>
#include <herring3/detail/forest.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>

namespace herring {
namespace detail {
namespace device_initialization {

template void initialize_device<
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >,
  kayak::device_type::gpu
>(kayak::device_id<kayak::device_type::gpu> device);

}
}
}
