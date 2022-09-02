#pragma once
#include <cstddef>
template<typename T>
struct matrix {
  T* data;
  std::size_t rows;
  std::size_t cols;
};
