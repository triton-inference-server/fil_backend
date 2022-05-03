#pragma once
#include <herring2/data_array.hpp>
#include <herring2/detail/index_type.hpp>
#include <herring2/flat_array.hpp>
#include <herring2/gpu_support.hpp>
#include <herring2/node_value.hpp>
#include <herring2/tree.hpp>
#include <herring2/tree_layout.hpp>

namespace herring {

template<tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, typename bitset_t>
struct forest {
  auto constexpr static const bounds_check = DEBUG_ENABLED && !GPU_ENABLED;
  using index_type = detail::index_type<bounds_check>;
  using category_set_type = bitset_t;
  using node_value_type = node_value<value_t, output_index_t, category_set_type>;
  using offset_type = offset_t;

  forest()
    : node_count_{}, values_{nullptr}, features_{nullptr},
    distant_offsets_{nullptr}, default_distant_{nullptr}, tree_count_{},
    tree_offsets_{nullptr}, output_size_{}, outputs_{nullptr},
    categorical_nodes_{nullptr}, node_categories_{nullptr} { }

  forest(
    index_type node_count,
    node_value_type* node_values,
    feature_index_t* node_features,
    offset_type* distant_child_offsets,
    bool* default_distant,
    index_type tree_count,
    raw_index_t* tree_offsets,
    index_type output_size = raw_index_t{1},
    output_t* outputs = nullptr,
    bool* categorical_nodes = nullptr,
    category_set_type* node_categories = nullptr
  ) : node_count_{node_count}, values_{node_values}, features_{node_features},
    distant_offsets_{distant_child_offsets}, default_distant_{default_distant}, tree_count_{tree_count},
    tree_offsets_{tree_offsets}, output_size_{output_size}, outputs_{outputs},
    categorical_nodes_{categorical_nodes}, node_categories_{node_categories} { }

  template <bool categorical, bool lookup, data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) const {
    // TODO(wphicks): host_only_throw if bounds_check enabled and tree index
    // OOB
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
  node_value_type* values_;
  feature_index_t* features_;
  offset_type* distant_offsets_;
  bool* default_distant_;

  raw_index_t tree_count_;
  raw_index_t* tree_offsets_;  // TODO(wphicks): Worth precomputing trees?

  raw_index_t output_size_;
  // Optional data (may be null)
  output_t* outputs_;
  bool* categorical_nodes_;
  category_set_type* node_categories_;

  template<bool lookup>
  HOST DEVICE auto get_output(index_type leaf_index) const {
    if constexpr (lookup) {
      return flat_array<array_encoding::dense, output_t>(
        outputs_ + leaf_index,
        output_size_
      );
    } else {
      auto const& value = values_ + leaf_index;
      if constexpr (std::is_same_v<value_t, output_t>) {
        return flat_array<array_encoding::dense, output_t>(
          &(value->value),
          1
        );
      } else if constexpr (std::is_same_v<value_t, output_t>) {
        return flat_array<array_encoding::dense, output_t>(
          &(value->index),
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
    // TODO(wphicks): Consider specialization for if tree is categorical
    auto tree = get_tree(tree_index);
    auto root_index_forest = tree_offsets_[tree_index];
    auto node_index_tree = raw_index_t{};
    auto offset = raw_index_t{};
    do {
      node_index_tree += offset;

      auto condition = false;
      if constexpr (categorical) {
        if (!categorical_nodes_[root_index_forest + node_index_tree]) {
          condition = evaluate_node<false>(
            root_index_forest + node_index_tree, row_index, input
          );
        } else {
          condition = evaluate_node<true>(
            root_index_forest + node_index_tree, row_index, input
          );
        }
      } else {
        condition = evaluate_node<false>(
          root_index_forest + node_index_tree, row_index, input
        );
      }

      offset = tree.next_offset(node_index_tree, condition);
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

      auto condition = false;
      if constexpr (categorical) {
        if (!categorical_nodes_[root_index_forest + node_index_tree]) {
          condition = evaluate_node<false>(
            root_index_forest + node_index_tree, row_index, input, missing_values
          );
        } else {
          condition = evaluate_node<true>(
            root_index_forest + node_index_tree, row_index, input, missing_values
          );
        }
      } else {
        condition = evaluate_node<false>(
          root_index_forest + node_index_tree, row_index, input, missing_values
        );
      }

      offset = tree.next_offset(node_index_tree, condition);
    } while (offset != raw_index_t{});

    return root_index_forest + node_index_tree;
  }

  HOST DEVICE [[nodiscard]] auto get_tree(index_type tree_index) const {
    auto min_index = tree_offsets_[tree_index];
    auto max_index = tree_index + 1 >= tree_count_ ? node_count_ : tree_offsets_[tree_index + 1];
    return tree<layout, offset_type>{distant_offsets_ + min_index, max_index - min_index};
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input
  ) const {
    auto result = false;
    auto value = input.at(row_index, features_[node_index]);
    if constexpr (!categorical) {
      result = value < values_[node_index].value;
    } else {
      if constexpr (sizeof(category_set_type) > sizeof(output_index_t)) {
        auto const& categories = node_categories_[values_[node_index].index];
        if (value >=0 && value < categories.size()) {
          // NOTE: This cast aligns with the convention used in LightGBM and
          // other frameworks to cast floats when converting to integral
          // categories. This can have surprising effects with floating point
          // arithmetic, but it is kept this way for now in order to provide
          // consistency with results obtained from the training frameworks.
          auto categorical_value = static_cast<raw_index_t>(value);
          // Too many categories for size of bitset storage; look up categories
          // in external storage
          result = categories.test(categorical_value);
        }
      } else {
        auto const& categories = category_set_type{values_[node_index].index};
        if (value >=0 && value < categories.size()) {
          // NOTE: This cast aligns with the convention used in LightGBM and
          // other frameworks to cast floats when converting to integral
          // categories. This can have surprising effects with floating point
          // arithmetic, but it is kept this way for now in order to provide
          // consistency with results obtained from the training frameworks.
          auto categorical_value = static_cast<raw_index_t>(value);
          // Too many categories for size of bitset storage; look up categories
          // in external storage
          result = categories.test(categorical_value);
        }
      }
    }
    return result;
  }

  template <bool categorical, data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    data_array<input_layout, input_t> const& input,
    data_array<input_layout, bool> const& missing_values
  ) const {
    auto col = features_[node_index];
    if (missing_values.at(row_index, col)) {
      return default_distant_[node_index];
    } else {
      return evaluate_node<categorical>(node_index, row_index, input);
    }
  }
};

}
