#pragma once
#include <herring2/detail/device_setter/base.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/device_setter/gpu.hpp>
#endif
#include <herring2/device_type.hpp>

namespace herring {

using device_setter = detail::device_setter<device_type::gpu>;

}
