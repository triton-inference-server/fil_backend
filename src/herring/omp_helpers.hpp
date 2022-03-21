#pragma once
#include <omp.h>

template <typename T>
struct thread_count {
  thread_count() : value{omp_get_max_threads()} {}
  thread_count(T t) : value{
    [](T t) {
      auto result = T{t};
      auto max_count = omp_get_max_threads();
      if ( t < 1 || t > max_count) {
        result = max_count;
      }
      return result;
    }(t)} {}
  operator int() const { return static_cast<int>(value); }
 private:
  T value;
};
