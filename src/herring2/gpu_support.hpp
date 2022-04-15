#pragma once

namespace herring {
#ifdef ENABLE_GPU
auto constexpr GPU_ENABLED = true;
#else
auto constexpr GPU_ENABLED = false;
#endif
}
