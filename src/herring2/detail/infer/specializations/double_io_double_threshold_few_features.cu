#include <herring2/detail/infer/specializations/gpu_predict.cuh>
#include <kayak/device_type.hpp>
#include <kayak/tree.hpp>

namespace herring {
namespace detail {
namespace inference {

template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, float, true>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, double, true>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint16_t, uint32_t, uint32_t, true>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, float, true>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, double, true>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, false>, double, double>;
template predict<kayak::device_type::gpu, forest<kayak::tree_layout::depth_first, double, uint16_t, uint32_t, uint32_t, uint32_t, true>, double, double>;

}
}
}
