#pragma once

namespace herring {

template<typename forest_t, typename output_t>
void infer(forest_t const& forest, std::optional<std::vector<output_t>> leaf_outputs) {
};

namespace detail {

// Relevant flags:
// categorical changes condition calculation
// missing_values changes condition calculation
// layout changes what we do with condition
template<typename forest_t, bool categorical_model>
void find_leaf(forest_t const& forest, std::size_t node_index) {
  auto offset = forest.node_offsets[node_index];
  auto condition = false;
  while(offset != 0) {
    auto node_value = forest.node_values[node_index];
    if constexpr (categorical_model) {
      if (!forest.categorical_nodes[node_index]) {
        condition = feature_value <= node_value.value;
      } else {
        if (feature_value >= 0 && feature_value < node.value.categories.size()) {
          // NOTE: This cast aligns with the convention used in LightGBM and
          // other frameworks to cast floats when converting to integral
          // categories. This can have surprising effects with floating point
          // arithmetic, but it is kept this way for now in order to provide
          // consistency with results obtained from the training frameworks.
          condition = node_value.categories[static_cast<std::size_t>(feature_value)];
        }
      }
    } else {
      condition = feature_value <= node_value.value;
    }
    node_index += typename forest_t::offset_type{1 + condition * (offset - 1)};
    offset = forest.node_offsets[node_index];
  }
  return node_index;
}

void find_leaf(forest_t const& forest, std::size_t node_index, bool missing_values) {
}

}

}
