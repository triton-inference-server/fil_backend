#pragma once
#include <herring2/data_array.hpp>
#include <herring2/detail/index_type.hpp>
#include <herring2/flat_array.hpp>
#include <herring2/gpu_support.hpp>
#include <herring2/node_value.hpp>
#include <herring2/tree.hpp>

namespace herring {

template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, typename bitset_t, tree_layout layout>
struct forest {
  using index_type = detail::index_type<DEBUG_ENABLED && !GPU_ENABLED>;

  template <bool categorical, bool lookup, data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) const {
    return get_output<lookup>(find_leaf<categorical>(tree_index, row_index, input));
  }

  template <bool categorical, bool lookup, data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input,
    data_array<input_layout, bool> const& missing_values
  ) const {
    return get_output<lookup>(
      find_leaf<categorical>(tree_index, row_index, input, missing_values)
    );
  }

 private:
  raw_index_t node_count_;
  offset_t* distant_offsets_;
  feature_index_t* features_;
  node_value<value_t, output_index_t, bitset_t>* values_;
  bool* default_distant_;

  raw_index_t tree_count_;
  raw_index_t* tree_offsets_;  // TODO(wphicks): Worth precomputing trees?

  raw_index_t output_size_;

  // Optional data (may be null)
  output_t* outputs_;
  bool* categorical_nodes_;
  bitset_t* node_categories_;

  template<bool lookup>
  HOST DEVICE auto get_output(index_type leaf_index) {
    if constexpr (lookup) {
      return flat_array<array_encoding::dense, output_t>(
        outputs_ + leaf_index,
        output_size_
      );
    } else {
      auto const& value = values_ + leaf_index;
      if constexpr (std::is_same_v<value_t, output_t>) {
        return flat_array<array_encoding::dense, output_t>(
          &(value.value),
          1
        );
      } else if constexpr (std::is_same_v<value_t, output_t>) {
        return flat_array<array_encoding::dense, output_t>(
          &(value.index),
          1
        );
      } else {
        // static_assert(false);
      }
    }
  }

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
        node_index_tree,
        evaluate_node<categorical>(root_index_forest + node_index_tree, row_index, input)
      );
    } while (offset != raw_index_t{});

    return root_index_forest + node_index_tree;
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
        node_index_tree,
        evaluate_node<categorical>(root_index_forest + node_index_tree, row_index, input, missing_values)
      );
    } while (offset != raw_index_t{});

    return root_index_forest + node_index_tree;
  }

  HOST DEVICE [[nodiscard]] auto get_tree(index_type tree_index) const {
    auto min_index = tree_offsets_[tree_index];
    auto max_index = tree_index + 1 >= tree_count_ ? node_count_ : tree_offsets_[tree_index + 1];
    return tree<layout, offset_t>{distant_offsets_ + min_index, max_index - min_index};
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) {
    if constexpr (!categorical) {
      return input.at(row_index, features_[node_index]) < values_[node_index].value;
    } else {
      auto const& categories = values_[node_index].categories;
      if constexpr (
        sizeof(bitset_t) > sizeof(value_t) && sizeof(bitset_t) > sizeof(output_index_t)
      ) {
        categories = node_categories_[values_[node_index].categories];
      }
      auto value = input.at(row_index, features_[node_index]);
      // TODO(wphicks): auto categorical_value = 
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
    if (missing_values.at(row_index, col)) {
      return default_distant_[node_index];
    } else {
      return evaluate_node(node_index, row_index, input);
    }
  }
};

}
