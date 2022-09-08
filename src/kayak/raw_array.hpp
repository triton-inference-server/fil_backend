#pragma once
#include <cstddef>

namespace kayak {
template<typename T, std::size_t N>
using raw_array = T[N];
}
