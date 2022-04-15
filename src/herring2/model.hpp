#pragma once
#include <bitset>
#include <cstddef>
#include <vector>
#include <herring2/forest_layout.hpp>

namespace herring {

template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t>
struct decision_forest {
  using value_type = value_t;  // float or double
  using index_type = feature_index_t;
  using offset_type = offset_t;
  using output_index_type = output_index_t;
  using output_type = output_t;
  using category_set_type = std::bitset<std::max(sizeof(value_type), sizeof(output_index_type))>;

  union value_or_index {
    value_type value;
    output_index_type index;
    // TODO(wphicks): Should categories be stored separately?
    category_set_type categories;
  };

  // Data
  // TODO(wphicks): These should all be allowed to live on GPU if desired
  std::vector<value_or_index> node_values;
  std::vector<index_type> node_features;
  std::vector<offset_type> node_offsets;
  std::vector<output_type> node_outputs;
  std::vector<bool> categorical_nodes;
  std::vector<bool> default_distant;
  // TODO(wphicks): Non-inclusive thresholds will be made inclusive via
  // next-representable trick

  // Metadata
  std::vector<std::size_t> tree_offsets;
  forest_layout layout;
  std::size_t features;
};

}
