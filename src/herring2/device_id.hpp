#pragma once

#include <herring2/detail/device_id/base.hpp>
#include <herring2/detail/device_id/cpu.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/device_id/gpu.hpp>
#endif
#include <herring2/device_type.hpp>
#include <variant>

namespace herring {
template <device_type D>
using device_id = detail::device_id<D>;

using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}
