#include <herring2/detail/infer/specializations/gpu_predict.cuh>
#include <kayak/device_type.hpp>
#include <kayak/tree.hpp>

namespace herring {
namespace detail {
namespace inference {

template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, true>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, true>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, true>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, true>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, true>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, false>, float, float>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, true>, float, float>;

}
}
}
