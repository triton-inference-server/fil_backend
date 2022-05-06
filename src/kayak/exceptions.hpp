#pragma once
#include <exception>

namespace kayak {
struct bad_cuda_call : std::exception {
  bad_cuda_call() : bad_cuda_call("CUDA API call failed") {}
  bad_cuda_call(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct out_of_bounds : std::exception {
  out_of_bounds() : out_of_bounds("Attempted out-of-bounds memory access") {}
  out_of_bounds(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
