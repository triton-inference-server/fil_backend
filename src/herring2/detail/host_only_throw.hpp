#pragma once
#include <herring2/detail/host_only_throw/base.hpp>
#include <herring2/detail/host_only_throw/cpu.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {
template<typename T, bool host=!GPU_COMPILATION>
using host_only_throw = detail::host_only_throw<T, host>;
}
