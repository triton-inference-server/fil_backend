#pragma once
#include <exception>

namespace herring {
#ifdef ENABLE_GPU
auto constexpr GPU_ENABLED = true;
#else
auto constexpr GPU_ENABLED = false;
#endif

struct gpu_unsupported : std::exception {
  gpu_unsupported() : gpu_unsupported("GPU functionality invoked in non-GPU build") {}
  gpu_unsupported(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
