#pragma once
#include <stdint.h>
#include <kayak/gpu_support.hpp>

namespace herring {
namespace detail {
namespace inference {

template <typename forest_t>
HOST DEVICE auto select_prediction_algorithm(forest_t const& forest) {
  auto categorical = forest.is_categorical();
  auto lookup = forest.requires_output_lookup();
  auto vector_leaf = forest.has_vector_leaves();
  return (
    (uint32_t{categorical} << 2u) +
    (uint32_t{lookup} << 1u) +
    uint32_t{vector_leaf}
  );
}

}
}
}
