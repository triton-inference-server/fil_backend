#pragma once
#include <herring2/detail/index_type.hpp>

namespace herring {
namespace detail {

/** C-style array alias for support on both device and host */
template<typename T, raw_index_t N>
using raw_array = T[N];

}
}
