#pragma once
#include <herring2/device_type.hpp>
#include <herring2/detail/owning_buffer/cpu.hpp>
#ifdef ENABLE_GPU
#include <herring2/detail/owning_buffer/gpu.hpp>
#endif
namespace herring {
template<device_type D, typename T>
using owning_buffer = detail::owning_buffer<D, T>;
}
