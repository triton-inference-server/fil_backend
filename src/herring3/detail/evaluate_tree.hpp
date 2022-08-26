#pragma once
#include <stdint.h>
#ifndef __CUDACC__
#include <math.h>
#endif
#include <kayak/bitset.hpp>
#include <kayak/gpu_support.hpp>
namespace herring {
namespace detail {

template<
  bool has_vector_leaves,
  bool has_categorical_nodes,
  typename node_t,
  typename io_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* __restrict__ node,
    io_t const* __restrict__ row
) {
  using categorical_set_type = kayak::bitset<uint32_t, typename node_t::index_type>;
  auto cur_node = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      if constexpr (has_categorical_nodes) {
        if (cur_node.is_categorical()) {
          auto valid_categories = categorical_set_type{
            &cur_node.index(),
            uint32_t(sizeof(typename node_t::index_type) * 8)
          };
          condition = valid_categories.test(input_val);
        } else {
          condition = (input_val < cur_node.threshold());
        }
      } else {
        condition = (input_val < cur_node.threshold());
      }
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  return cur_node.template output<has_vector_leaves>();
}

template<
  bool has_vector_leaves,
  typename node_t,
  typename io_t,
  typename categorical_storage_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* __restrict__ node,
    io_t const* __restrict__ row,
    categorical_storage_t const* __restrict__ categorical_storage
) {
  using categorical_set_type = kayak::bitset<uint32_t, categorical_storage_t>;
  auto cur_node = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      if (cur_node.is_categorical()) {
        auto valid_categories = categorical_set_type{
          categorical_storage + cur_node.index() + 1,
          uint32_t(categorical_storage[cur_node.index()])
        };
        condition = valid_categories.test(input_val);
      } else {
        condition = (input_val < cur_node.threshold());
      }
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  return cur_node.template output<has_vector_leaves>();
}

}
}
