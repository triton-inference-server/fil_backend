#pragma once
#include <stdint.h>
#ifndef __CUDACC__
#include <math.h>
#endif
#include <kayak/bitset.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/raw_array.hpp>
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
  using categorical_set_type = kayak::bitset<uint32_t, typename node_t::index_type const>;
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
  bool has_categorical_nodes,
  uint8_t simultaneous_rows,
  typename node_t,
  typename io_t,
  typename leaf_out_t
>
HOST DEVICE auto evaluate_tree(
    node_t const* node,
    io_t const* __restrict__ row,
    leaf_out_t* out,
    uint8_t num_rows,
    size_t num_cols
) {
  using categorical_set_type = kayak::bitset<uint32_t, typename node_t::index_type const>;
  auto nodes = kayak::raw_array<node_t const*, simultaneous_rows>{};
  auto* result = out;
// #pragma unroll
  for (auto i=uint8_t{}; i < simultaneous_rows; ++i) {
    nodes[i] = node;
  }
  uint8_t active_rows_mask = ~(1 << num_rows);
  do {
// #pragma unroll
    for (auto i=uint8_t{}; i < simultaneous_rows; ++i) {
      active_rows_mask &= ~(nodes[i]->is_leaf() << i);
      if ((active_rows_mask & (uint8_t{1} << i)) != uint8_t{}) {
        auto input_val = row[nodes[i]->feature_index()];
        auto condition = nodes[i]->default_distant();
        if (!isnan(input_val)) {
          if constexpr (has_categorical_nodes) {
            if (nodes[i]->is_categorical()) {
              auto valid_categories = categorical_set_type{
                &(nodes[i]->index()),
                uint32_t(sizeof(typename node_t::index_type) * 8)
              };
              condition = valid_categories.test(input_val);
            } else {
              condition = (input_val < nodes[i]->threshold());
            }
          } else {
            condition = (input_val < nodes[i]->threshold());
          }
        }
        nodes[i] += nodes[i]->child_offset(condition);
      }
    }
  } while (active_rows_mask != uint8_t{});
// #pragma unroll
  for (auto i=uint8_t{}; i < simultaneous_rows; ++i) {
    result[i] = nodes[i]->template output<has_vector_leaves>();
  }
  return result;
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
  using categorical_set_type = kayak::bitset<uint32_t, categorical_storage_t const>;
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
