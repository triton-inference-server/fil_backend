#pragma once
#include <herring2/device_type.hpp>
#include <herring2/detail/non_owning_buffer/base.hpp>

namespace herring {
template<device_type D, typename T>
using non_owning_buffer = detail::non_owning_buffer<D, T>;
}
