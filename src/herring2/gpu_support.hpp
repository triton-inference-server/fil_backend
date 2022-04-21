#pragma once
#include <exception>
#include <cstddef>
#include <stdint.h>

namespace herring {
#ifdef ENABLE_GPU
auto constexpr GPU_ENABLED = true;
#else
auto constexpr GPU_ENABLED = false;
#endif

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
using index_t = uint32_t;
using diff_t = int32_t;
#else
#define HOST
#define DEVICE
#define GLOBAL
using index_t = std::size_t;
using diff_t = std::ptrdiff_t;
#endif

struct gpu_unsupported : std::exception {
  gpu_unsupported() : gpu_unsupported("GPU functionality invoked in non-GPU build") {}
  gpu_unsupported(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
