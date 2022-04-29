#pragma once
#include <herring2/data_array.hpp>
#include <herring2/detail/index_type.hpp>
#include <herring2/flat_array.hpp>
#include <herring2/gpu_support.hpp>
#include <herring2/tree.hpp>

namespace herring {

template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, tree_layout layout>
struct forest {
  using index_type = detail::index_type<DEBUG_ENABLED && !GPU_ENABLED>;

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) const {
    return flat_array<array_encoding::dense>(
      outputs_ + find_leaf<categorical>(tree_index, row_index, input),
      output_size_
    );
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input,
    data_array<input_layout, bool> const& missing_values
  ) const {
    return flat_array<array_encoding::dense>(
      outputs_ + find_leaf<categorical>(tree_index, row_index, input, missing_values),
      output_size_
    );
  }

 private:
  raw_index_t node_count_;
  offset_t* distant_offsets_;
  feature_index_t* features_;
  value_or_index* values_;
  bool* default_distant_;

  raw_index_t tree_count_;
  raw_index_t* tree_offsets_;  // TODO(wphicks): Worth precomputing trees?

  raw_index_t output_size_;

  // Optional data (may be null)
  output_t* outputs_;
  bool* categorical_nodes_;
  // TODO(wphicks): Add optional separate category storage for many categories

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE auto find_leaf(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) const {
    auto tree = get_tree(tree_index);
    auto root_index_forest = tree_offsets_[tree_index];
    auto node_index_tree = raw_index_t{};
    auto offset = raw_index_t{};
    do {
      node_index_tree += offset;
      offset = tree.next_offset(
        node_tree_index,
        evaluate_node<categorical>(root_index_forest + node_index_tree, row_index, input)
      );
    } while (offset != raw_index_t{});

    return root_index_forest + tree_node;
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE auto find_leaf(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input,
    data_array<input_layout, bool> const& missing_values
  ) const {
    auto tree = get_tree(tree_index);
    auto root_index_forest = tree_offsets_[tree_index];
    auto node_index_tree = raw_index_t{};
    auto offset = raw_index_t{};
    do {
      node_index_tree += offset;
      offset = tree.next_offset(
        node_tree_index,
        evaluate_node<categorical>(root_index_forest + node_index_tree, row_index, input, missing_values)
      );
    } while (offset != raw_index_t{});

    return root_index_forest + tree_node;
  }

  HOST DEVICE [[nodiscard]] auto get_tree(index_type tree_index) const {
    auto min_index = tree_offsets_[tree_index];
    auto max_index = tree_index + 1 >= tree_counts ? node_count : tree_offsets_[tree_index + 1];
    return tree<layout, offset_t>{distant_offsets + min_index, max_index - min_index};
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) {
    if constexpr (!categorical) {
      return input.at(row, features_[node_index]) < values_[node_index].value;
    } else {
      // TODO(wphicks): Add categorical logic
    }
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input,
    data_array<input_layout, bool> const& missing_values
  ) {
    auto col = features_[node_index];
    if (missing_values.at(row, col)) {
      return default_distant_[node_index];
    } else {
      return evaluate_node(node_index, row_index, input);
    }
  }
};
