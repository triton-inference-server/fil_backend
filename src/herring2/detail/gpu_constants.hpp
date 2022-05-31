#pragma once

namespace herring {
namespace detail {
namespace inference {
namespace gpu {

auto constexpr static const WARP_SIZE = 32u;
auto constexpr static const LINE_SIZE = 128u;
auto constexpr static const CHUNK_SIZE = LINE_SIZE / WARP_SIZE;
auto constexpr static const INV_CHUNK_SIZE = double{WARP_SIZE} / LINE_SIZE;
auto constexpr static const INV_WARP_SIZE = 1.0 / WARP_SIZE;

}
}
}
}
