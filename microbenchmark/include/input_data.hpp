#pragma once
#include <kayak/buffer.hpp>

template <typename T>
struct input_data {
  // Note: This whole struct should be replaced with an mdarray, but currently
  // the mdarray headers cannot be used in a CUDA-free build
  kayak::buffer<T> data;
  std::size_t rows;
  std::size_t cols;
};
