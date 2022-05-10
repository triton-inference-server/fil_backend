#pragma once
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <exception>
#include <herring2/forest.hpp>
#include <kayak/bitset.hpp>
#include <kayak/buffer.hpp>
#include <herring2/node_value.hpp>
#include <kayak/tree.hpp>
#include <kayak/tree_layout.hpp>
#include <limits>
#include <optional>
#include <variant>

namespace herring {

struct unusable_model_exception : std::exception {
  unusable_model_exception () : msg_{"Model is not compatible with Herring"}
  {
  }
  unusable_model_exception (std::string msg) : msg_{msg}
  {
  }
  unusable_model_exception (char const* msg) : msg_{msg}
  {
  }
  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

template<kayak::tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, bool categorical_lookup>
struct decision_forest {
  using forest_type = forest<
    layout,
    value_t,
    feature_index_t,
    offset_t,
    output_index_t,
    output_t,
    categorical_lookup
  >;
  using value_type = typename forest_type::value_type;
  using feature_index_type = typename forest_type::feature_index_type;
  using offset_type = typename forest_type::offset_type;
  using output_index_type = typename forest_type::output_index_type;
  using output_type = typename forest_type::output_type;
  using node_value_type = typename forest_type::node_value_type;
  using category_set_type = typename forest_type::category_set_type;

  auto num_features() const { return num_features_; }
  auto outputs_per_sample() const { return output_size_; }

  auto obj() const {
    auto node_output_ptr = static_cast<output_type*>(nullptr);
    if (node_outputs) {
      node_output_ptr = node_outputs->buffer().data();
    }
    auto categorical_sizes_ptr = static_cast<raw_index_t*>(nullptr);
    if (categorical_sizes) {
      categorical_sizes_ptr = categorical_sizes->buffer().data();
    }
    auto categorical_storage_ptr = static_cast<uint8_t*>(nullptr);
    if (categorical_storage) {
      categorical_storage_ptr = categorical_storage->data();
    }
    return forest{
      node_offsets.objs(),
      node_values.buffer().size(),
      node_features.buffer().data(),
      default_distant.buffer().data(),
      output_size_,
      node_output_ptr,
      categorical_sizes_ptr,
      categorical_storage_ptr
    };
  }

 private:
  // Data
  kayak::multi_tree<layout, offset_type> node_offsets;
  kayak::multi_flat_array<kayak::array_encoding::dense, node_value_type> node_values;
  kayak::multi_flat_array<kayak::array_encoding::dense, feature_index_type> node_features;
  kayak::multi_flat_array<kayak::array_encoding::dense, bool> default_distant;
  std::optional<kayak::multi_flat_array<kayak::array_encoding::dense, output_type>> node_outputs;
  std::optional<kayak::multi_flat_array<kayak::array_encoding::dense, raw_index_t>> categorical_sizes;
  std::optional<kayak::buffer<uint8_t>> categorical_storage;
  // TODO(wphicks): Non-inclusive thresholds will be made inclusive via
  // next-representable trick

  // Metadata
  std::size_t num_class_;
  std::size_t num_features_;
  std::size_t output_size_;
};

template<
  typename output_t,
  kayak::tree_layout layout,
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
  many_categories
>;

namespace detail {
auto constexpr static const preferred_tree_layout = kayak::tree_layout::depth_first;
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
  std::size_t max_nodes_per_tree,
  std::size_t max_depth,
  std::size_t num_features,
  std::size_t max_num_categories,
  bool use_double_thresholds
) {
  auto constexpr small_value = std::size_t{0};
  auto constexpr large_value = std::size_t{1};

  auto constexpr precision_bit = std::size_t{3};

  auto constexpr features_bit = std::size_t{2};
  auto constexpr max_few_features = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << features_bit), forest_model_variant<output_t>
  >::feature_index_type>::max();
  auto constexpr max_many_features = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << features_bit), forest_model_variant<output_t>
  >::feature_index_type>::max();

  auto constexpr tree_bit = std::size_t{1};
  auto constexpr max_small_trees = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << tree_bit), forest_model_variant<output_t>
  >::offset_type>::max();
  auto constexpr max_large_trees = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << tree_bit), forest_model_variant<output_t>
  >::offset_type>::max();

  auto constexpr category_bit = std::size_t{0};
  auto constexpr max_few_categories = std::variant_alternative_t<
    (small_value << category_bit), forest_model_variant<output_t>
  >::category_set_type::size();
  auto constexpr max_many_categories = std::numeric_limits<raw_index_t>::max();

  if (num_features > max_many_features) {
    throw unusable_model_exception("Model contains too many features");
  }

  auto max_node_offset = std::min(max_nodes_per_tree, (std::size_t{1} << max_depth));
  if (max_node_offset > max_large_trees) {
    throw unusable_model_exception("Model contains too large of trees");
  }

  if (max_num_categories > max_many_categories) {
    throw unusable_model_exception("Model contains feature with too many categories");
  }

  auto has_many_categories = std::size_t{max_num_categories > max_few_categories};
  auto has_large_trees = std::size_t{max_node_offset > max_small_trees};
  auto has_many_features = std::size_t{num_features > max_few_features};
  
  return std::size_t{
    (std::size_t{use_double_thresholds} << precision_bit) +
    (has_many_features << features_bit) +
    (has_large_trees << tree_bit) +
    (has_many_categories << category_bit)
  };
}
}
