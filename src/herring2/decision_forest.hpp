#pragma once
#include <bitset>
#include <cstddef>
#include <functional>
#include <optional>
#include <herring/output_ops.hpp>
#include <herring/type_helpers.hpp>
#include <herring2/forest_layout.hpp>
#include <herring2/buffer.hpp>

namespace herring {

template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t>
struct decision_forest {
  using value_type = value_t;  // float or double
  using index_type = feature_index_t;
  using offset_type = offset_t;
  using output_index_type = output_index_t;
  using output_type = output_t;
  using category_set_type = std::bitset<std::max(sizeof(value_type), sizeof(output_index_type))>;
  using sum_elem_type = typename is_container_specialization<output_t, buffer>::value_type;

  union value_or_index {
    value_type value;
    output_index_type index;
    // TODO(wphicks): Should categories be stored separately?
    category_set_type categories;
  };

  // Data
  buffer<value_or_index> node_values;
  buffer<index_type> node_features;
  buffer<offset_type> node_offsets;
  buffer<bool> default_distant;
  std::optional<buffer<output_type>> node_outputs;
  std::optional<buffer<bool>> categorical_nodes;
  // TODO(wphicks): Non-inclusive thresholds will be made inclusive via
  // next-representable trick

  // Metadata
  buffer<std::size_t> tree_offsets;
  forest_layout layout;
  std::size_t num_class;
};

}
