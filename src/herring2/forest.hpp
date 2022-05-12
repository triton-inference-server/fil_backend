#pragma once
#include <iostream>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/bitset.hpp>
#include <kayak/flat_array.hpp>
#include <kayak/gpu_support.hpp>
#include <herring2/node_value.hpp>
#include <kayak/tree.hpp>
#include <kayak/tree_layout.hpp>

namespace herring {

using kayak::raw_index_t;

template<kayak::tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, bool categorical_lookup>
struct forest {
  auto constexpr static const bounds_check = kayak::DEBUG_ENABLED && !kayak::GPU_ENABLED;
  auto constexpr static const tree_layout = layout;
  using index_type = kayak::detail::index_type<bounds_check>;
  using category_set_type = std::conditional_t<
    categorical_lookup,
    kayak::bitset<uint8_t>,
    kayak::bitset<output_index_t>
  >;
  using node_value_type = node_value<value_t, output_index_t, category_set_type>;
  using offset_type = offset_t;
  using output_index_type = output_index_t;

  forest()
    : trees_{}, values_{nullptr}, features_{nullptr},
    default_distant_{nullptr},
    output_size_{raw_index_t{1}}, outputs_{nullptr},
    categorical_sizes_{nullptr}, categorical_storage_{nullptr} { }

  forest(
    kayak::flat_array<kayak::array_encoding::dense, kayak::tree<layout, offset_type>> const& trees,
    node_value_type* node_values,
    feature_index_t* node_features,
    bool* default_distant,
    index_type output_size = raw_index_t{1},
    output_t* outputs = nullptr,
    raw_index_t* categorical_sizes = nullptr,
    uint8_t* categorical_storage = nullptr
  ) : trees_{trees}, values_{node_values}, features_{node_features},
    default_distant_{default_distant},
    output_size_{output_size}, outputs_{outputs},
    categorical_sizes_{categorical_sizes}, categorical_storage_{categorical_storage} { }

  template <bool categorical, bool lookup, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input
  ) const {
    // TODO(wphicks): host_only_throw if bounds_check enabled and tree index
    // OOB
    return get_output<lookup>(find_leaf<categorical>(tree_index, row_index, input));
  }

  template <bool categorical, bool lookup, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE auto evaluate_tree(
    index_type tree_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input,
    kayak::data_array<input_layout, bool> const& missing_values
  ) const {
    return get_output<lookup>(
      find_leaf<categorical>(tree_index, row_index, input, missing_values)
    );
  }

 private:
  kayak::flat_array<kayak::array_encoding::dense, kayak::tree<layout, offset_type>> trees_;
  kayak::flat_array<kayak::array_encoding::dense, node_value_type> values_;
  kayak::flat_array<kayak::array_encoding::dense, feature_index_t> features_;
  kayak::flat_array<kayak::array_encoding::dense, bool> default_distant_;

  raw_index_t tree_count_;

  raw_index_t output_size_;
  // Optional data (may be size 0)
  kayak::flat_array<kayak::array_encoding::dense, output_t> outputs_;
  kayak::flat_array<kayak::array_encoding::dense, raw_index_t> categorical_sizes_;
  kayak::flat_array<kayak::array_encoding::dense, uint8_t> categorical_storage_;

  template<bool lookup>
  HOST DEVICE auto get_output(index_type leaf_index) const {
    if constexpr (lookup) {
      return kayak::flat_array<kayak::array_encoding::dense, output_t const>(
        outputs_ + values_[leaf_index].index,
        output_size_
      );
    } else {
      auto const& value = values_[leaf_index];
      if constexpr (std::is_same_v<value_t, output_t>) {
        return kayak::flat_array<kayak::array_encoding::dense, output_t const>(
          &(value.value),
          1
        );
      } else if constexpr (std::is_same_v<output_index_t, output_t>) {
        return kayak::flat_array<kayak::array_encoding::dense, output_t const>(
          &(value.index),
          1
        );
      } else {
        static_assert(lookup);
      }
    }
  }

  template <bool categorical, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE auto find_leaf(
    index_type tree_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input
  ) const {
    // TODO(wphicks): Consider specialization for if tree is categorical
    auto tree = get_tree(tree_index);
    auto root_index_forest = tree.data() - trees_[0].data();
    auto node_index_tree = raw_index_t{};
    auto offset = raw_index_t{};

    std::cout << tree_index << " -> ";

    while (tree[node_index_tree] != offset_type{}) {
      auto condition = false;
      std::cout << node_index_tree << ": ";
      if constexpr (categorical) {
        if (categorical_sizes_[root_index_forest + node_index_tree] == 0) {
          condition = evaluate_node<false>(
            root_index_forest + node_index_tree, row_index, input
          );
          std::cout << " non-categorical ";
        } else {
          condition = evaluate_node<true>(
            root_index_forest + node_index_tree, row_index, input
          );
          std::cout << " categorical ";
        }
      } else {
        condition = evaluate_node<false>(
          root_index_forest + node_index_tree, row_index, input
        );
        std::cout << " categorical-free ";
      }
      if (condition) {
        std::cout << "YES, ";
      } else {
        std::cout << "NO, ";
      }
      if constexpr (layout == kayak::tree_layout::depth_first) {
        offset = 1 + (tree[node_index_tree] - 1) * condition;
      } else if constexpr (layout == kayak::tree_layout::breadth_first) {
        offset = tree[node_index_tree] + condition - 1;
      } else {
        static_assert(layout == kayak::tree_layout::depth_first);
      }
      node_index_tree += offset;
    }
    std::cout << "\n";

    return root_index_forest + node_index_tree;
  }

  template <bool categorical, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE auto find_leaf(
    index_type tree_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input,
    kayak::data_array<input_layout, bool> const& missing_values
  ) const {
    auto tree = get_tree(tree_index);
    auto root_index_forest = tree.data() - trees_[0].data();
    auto node_index_tree = raw_index_t{};
    auto offset = raw_index_t{};

    while (tree[node_index_tree] != offset_type{}) {
      auto condition = false;
      if constexpr (categorical) {
        if (categorical_sizes_[root_index_forest + node_index_tree] == 0) {
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
      if constexpr (layout == kayak::tree_layout::depth_first) {
        offset = 1 + (tree[node_index_tree] - 1) * condition;
      } else if constexpr (layout == kayak::tree_layout::breadth_first) {
        offset = tree[node_index_tree] + condition - 1;
      } else {
        static_assert(layout == kayak::tree_layout::depth_first);
      }
      node_index_tree += offset;
    }

    return root_index_forest + node_index_tree;
  }

  HOST DEVICE [[nodiscard]] auto get_tree(index_type tree_index) const {
    return trees_.at(tree_index);
  }

  template <bool categorical, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input
  ) const {
    auto result = false;
    auto value = input.at(row_index, features_[node_index]);
    if constexpr (!categorical) {
      result = value < values_[node_index].value;
    } else {
      auto categories = category_set_type{};
      if constexpr (categorical_lookup) {
        categories = category_set_type{
          categorical_storage_ + values_[node_index].index,
          categorical_sizes_[node_index]
        };
      } else {
        categories = category_set_type{&(values_[node_index].index)};
      }
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
    return result;
  }

  template <bool categorical, kayak::data_layout input_layout, typename input_t>
  HOST DEVICE [[nodiscard]] auto evaluate_node(
    index_type node_index,
    index_type row_index,
    kayak::data_array<input_layout, input_t> const& input,
    kayak::data_array<input_layout, bool> const& missing_values
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
