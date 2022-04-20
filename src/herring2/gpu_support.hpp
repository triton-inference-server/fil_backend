#pragma once
#include <exception>

namespace herring {
#ifdef ENABLE_GPU
auto constexpr GPU_ENABLED = true;
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#else
auto constexpr GPU_ENABLED = false;
#define HOST
#define DEVICE
#define GLOBAL
#endif

struct gpu_unsupported : std::exception {
  gpu_unsupported() : gpu_unsupported("GPU functionality invoked in non-GPU build") {}
  gpu_unsupported(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
