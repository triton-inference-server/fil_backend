#pragma once
#include <new>
#include <kayak/detail/index_type.hpp>

namespace herring {
namespace detail {
namespace inference {
namespace cpu {

#ifdef __cpp_lib_hardware_interference_size
  using std::hardware_constructive_interference_size;
#else
  auto constexpr hardware_constructive_interference_size = std::size_t{64};
#endif

auto constexpr static const LINE_SIZE = kayak::detail::index_type<false>{hardware_constructive_interference_size}.value();
auto constexpr static const CHUNK_SIZE = LINE_SIZE;
auto constexpr static const GROVE_SIZE = LINE_SIZE;

}
}
}
}
