#pragma once
#ifndef __CUDACC__
#include <math.h>
#endif
#include <kayak/gpu_support.hpp>
namespace herring {
namespace detail {

template<
  typename leaf_output_t,
  typename node_t,
  typename io_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* __restrict__ node,
    io_t const* __restrict__ row
) {
  auto cur_node = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      condition = (input_val < cur_node.threshold());
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  return cur_node.template output<leaf_output_t>();
}

}
}
