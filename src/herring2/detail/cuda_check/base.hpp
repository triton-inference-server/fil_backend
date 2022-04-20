#pragma once
#include <herring2/device_type.hpp>

namespace herring {
namespace detail {

template <device_type D, typename error_t>
void cuda_check(error_t const& err) {
}

}
}
