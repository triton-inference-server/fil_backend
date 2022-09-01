#include <cstddef>
#include <herring3/constants.hpp>
#include <herring3/detail/forest.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <herring3/detail/infer/cpu.hpp>
#include <herring3/detail/specialization_macros.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <kayak/tree_layout.hpp>
namespace herring {
namespace detail {
namespace inference {
HERRING_INFER_ALL(template, kayak::device_type::cpu, 0)
HERRING_INFER_ALL(template, kayak::device_type::cpu, 1)
HERRING_INFER_ALL(template, kayak::device_type::cpu, 2)
HERRING_INFER_ALL(template, kayak::device_type::cpu, 3)
}
}
}
