#pragma once
#include <stdint.h>
#include <cstddef>
#include <herring/output_ops.hpp>
#include <herring/type_helpers.hpp>
#include <herring2/bitset.hpp>
#include <herring2/buffer.hpp>
#include <herring2/node_value.hpp>
#include <herring2/tree_layout.hpp>
#include <limits>
#include <optional>
#include <variant>

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
  using feature_index_type = feature_index_t;
  using offset_type = offset_t;
  using output_index_type = output_index_t;
  using output_type = output_t;
  using category_set_type = categories_t;
  using node_value_type = node_value<value_type, output_index_type, category_set_type>;

  // Data
  buffer<node_value_type> node_values;
  buffer<feature_index_type> node_features;
  buffer<offset_type> node_offsets;
  buffer<bool> default_distant;
  std::optional<buffer<output_type>> node_outputs;
  std::optional<buffer<bool>> categorical_nodes;
  // TODO(wphicks): Non-inclusive thresholds will be made inclusive via
  // next-representable trick

  // Metadata
  buffer<std::size_t> tree_offsets;
  std::size_t num_class;
};

template<
  typename output_t,
  tree_layout layout,
  bool double_precision,
  bool many_features,
  bool large_trees,
  bool many_categories
>
using forest_model = decision_forest<
  layout,
  std::conditional_t<double_precision, double, float>,
  std::conditional_t<many_features, uint32_t, uint16_t>,
  std::conditional_t<large_trees, uint32_t, uint16_t>,
  uint32_t,
  output_t,
  std::conditional_t<
    many_categories,
    bitset<32, uint32_t>,
    bitset<1024, uint8_t>
  >
>;

namespace detail {
auto constexpr static const preferred_tree_layout = tree_layout::depth_first;
}

template <typename output_t>
using forest_model_variant = std::variant<
  forest_model<output_t, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<output_t, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<output_t, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<output_t, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<output_t, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<output_t, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<output_t, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<output_t, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<output_t, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<output_t, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<output_t, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<output_t, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<output_t, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<output_t, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<output_t, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<output_t, detail::preferred_tree_layout, true, true, true, true>
>;

template <typename output_t>
auto get_forest_variant_index(
  std::size_t num_nodes,
  std::size_t max_depth,
  std::size_t num_features,
  std::size_t num_categories,
  bool use_double_thresholds
) {
  auto constexpr small_value = std::size_t{1};
  auto constexpr large_value = std::size_t{1};

  auto constexpr category_bit = std::size_t{0};
  auto constexpr max_few_categories = std::variant_alternative_t<
    (small_value << category_bit), forest_model_variant<output_t>
  >::category_set_type::size();
  auto constexpr max_many_categories = std::variant_alternative_t<
    (large_value << category_bit), forest_model_variant<output_t>
  >::category_set_type::size();

  auto constexpr tree_bit = std::size_t{1};
  auto constexpr max_small_trees = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << tree_bit), forest_model_variant<output_t>
  >::offset_type>::max();
  auto constexpr max_large_trees = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << tree_bit), forest_model_variant<output_t>
  >::offset_type>::max();

  auto constexpr features_bit = std::size_t{1};
  auto constexpr max_few_features = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << features_bit), forest_model_variant<output_t>
  >::feature_index_type>::max();
  auto constexpr max_many_features = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << features_bit), forest_model_variant<output_t>
  >::feature_index_type>::max();

  if (num_categories > max_many_categories) {
    throw unusable_model("Model contains too many categorical values in a single category");
  }
  if (num_features > max_many_features) {
    throw unusable_model("Model contains too many features");
  }
}
}
