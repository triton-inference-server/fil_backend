#pragma once
#include <bitset>
#include <cstddef>
#include <functional>
#include <optional>
#include <herring/output_ops.hpp>
#include <herring/type_helpers.hpp>
#include <herring2/buffer.hpp>
#include <herring2/node_value.hpp>
#include <herring2/tree_layout.hpp>

namespace herring {

/* value_t: float/double
 * feature_index_t: uint16_t or uint32_t
 * offset_t: uint16_t or uint32_t
 * output_index_t: uint32_t
 * output_t: float, double, or uint32_t
 * categories_t: 32, 1024
 */

template<tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, typename categories_t>
struct decision_forest {
  using value_type = value_t;  // float or double
  using index_type = feature_index_t;
  using offset_type = offset_t;
  using output_index_type = output_index_t;
  using output_type = output_t;
  using category_set_type = categories_t;
  using node_value_type = node_value<value_type, output_index_type, category_set_type>;

  // Data
  buffer<node_value_type> node_values;
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
