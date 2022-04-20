#pragma once
#include <optional>

namespace herring {

template<typename forest_t, typename input_array_t, typename output_array_t>
void infer(forest_t const& forest, input_array_t const& input, output_array_t const& output) {
};

namespace detail {

/* Return whether the node at node_index in forest returns true or false for
 * the row at row_index in input
 */
template<typename forest_t, typename input_array_t>
auto evaluate_node(forest_t const& forest, input_array_t const& input, std::size_t node_index, std::size_t row_index) {
  auto result = false;
  auto node_value = forest.node_values[node_index];
  auto feature_value = input.get_value(row_index, forest.node_features[node_index])

  if constexpr (categorical_tree) {
    // TODO(wphicks): Try eliminating this branch
    if (!forest.categorical_nodes[node_index]) {
      result = feature_value <= node_value.value;
    } else {
      if (feature_value >= 0 && feature_value < node.value.categories.size()) {
        // NOTE: This cast aligns with the convention used in LightGBM and
        // other frameworks to cast floats when converting to integral
        // categories. This can have surprising effects with floating point
        // arithmetic, but it is kept this way for now in order to provide
        // consistency with results obtained from the training frameworks.
        result = node_value.categories[static_cast<std::size_t>(feature_value)];
      }
    }
  } else {
    result = feature_value <= node_value.value;
  }
  return result;
}

/* Return the index of the leaf node obtained by evaluating the decision tree
 * beginning at the node indicated by node_index for the row indicated by
 * row_index in input, assuming no missing values in that row
 */
template<bool categorical_tree, typename forest_t, typename input_array_t>
auto find_leaf(
    forest_t const& forest,
    input_array_t const& input,
    std::size_t node_index,
    std::size_t row_index) {
  auto offset = forest.node_offsets[node_index];
  auto condition = false;
  while(offset != 0) {
    condition = evaluate_node(forest, input, node_index, row_index);
    node_index += typename forest_t::offset_type{1 + condition * (offset - 1)};
    offset = forest.node_offsets[node_index];
  }
  return node_index;
}

/* Return the index of the leaf node obtained by evaluating the decision tree
 * beginning at the node indicated by node_index for the row indicated by
 * row_index in input, with missing values for input indicated by
 * missing_values
 */
template<bool categorical_tree, typename forest_t, typename input_array_t>
auto find_leaf(
    forest_t const& forest,
    input_array_t const& input,
    std::size_t node_index,
    std::size_t row_index,
    std::vector<bool> const& missing_values) {

  auto offset = forest.node_offsets[node_index];
  auto condition = false;
  while(offset != 0) {
    if (missing_values[input.get_index(row_index, forest.node_features[node_index])]) {
      condition = forest.default_distant[node_index];
    } else {
      condition = evaluate_node(forest, input, node_index, row_index);
    }
    node_index += typename forest_t::offset_type{1 + condition * (offset - 1)};
    offset = forest.node_offsets[node_index];
  }
  return node_index;
}

/* Return the output value associated with the node of forest at node_index
 */
template<typename forest_t, typename output_t>
auto get_leaf_output(forest_t const& forest, std::size_t node_index) {
  auto node_value = forest.node_values[node_index];
  if constexpr (std::is_same_v<output_t, typename forest_t::output_index_type>) {
    return node_value.value;
  } else if constexpr (std::is_same_v<output_t, typename forest_t::output_index_type>) {
    return node_value.index;
  } else {
    return forest.node_outputs[node_value.index];
  }
}

template<typename forest_t, typename input_array_t, std::size_t grove_size, std::size_t block_size>
auto get_grove_output(forest_t const& forest, input_array_t const& input) {
  auto const num_tree = forest.tree_offsets.size();
  auto const num_grove = (num_tree / grove_size + (num_tree % grove_size != 0));
  auto const num_block = (num_row / block_size + (num_row % block_size != 0));

  using sum_elem_type = typename forest_t::sum_elem_type;
  auto result = std::vector<sum_elem_type>(
    num_row * forest.num_class * num_grove,
    sum_elem_type{}
  );
}

} // namespace detail

} // namespace herring
