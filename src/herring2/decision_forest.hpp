#pragma once
#include <nvtx3/nvtx3.hpp>
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <herring/output_ops.hpp>
#include <herring2/detail/infer.hpp>
#include <herring2/detail/preferred_tree_layout.hpp>
#include <herring2/exceptions.hpp>
#include <herring2/forest.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/tree_layout.hpp>
#include <limits>
#include <optional>
#include <variant>

namespace herring {

template<kayak::tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, bool categorical_lookup_v>
struct decision_forest {
  auto constexpr static const categorical_lookup = categorical_lookup_v;
  using forest_type = forest<
    layout,
    value_t,
    feature_index_t,
    offset_t,
    output_index_t,
    output_t,
    categorical_lookup
  >;
  using index_type = typename forest_type::index_type;
  using feature_index_type = typename forest_type::feature_index_type;
  using offset_type = typename forest_type::offset_type;
  using output_index_type = typename forest_type::output_index_type;
  using output_type = typename forest_type::output_type;
  using node_value_type = typename forest_type::node_value_type;
  using category_set_type = typename forest_type::category_set_type;


  auto num_features() const { return num_features_; }
  auto leaf_size() const { return leaf_size_; }

  auto obj() const {
    auto node_output_ptr = static_cast<output_type*>(nullptr);
    if (node_outputs_) {
      node_output_ptr = node_outputs_->data();
    }
    auto categorical_sizes_ptr = static_cast<raw_index_t*>(nullptr);
    if (categorical_sizes_) {
      categorical_sizes_ptr = categorical_sizes_->data();
    }
    auto categorical_storage_ptr = static_cast<uint8_t*>(nullptr);
    if (categorical_storage_) {
      categorical_storage_ptr = categorical_storage_->data();
    }
    return forest_type{
      node_values_.size(),
      node_values_.data(),
      node_features_.data(),
      node_offsets_.data(),
      default_distant_.data(),
      tree_offsets_.size(),
      tree_offsets_.data(),
      leaf_size_,
      node_output_ptr,
      categorical_sizes_ptr,
      categorical_storage_ptr
    };
  }

  auto memory_type() { return tree_offsets_.memory_type(); }

  template<typename io_t>
  void predict(
    kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
    kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
    int device_id=0,
    kayak::cuda_stream stream=kayak::cuda_stream{}
  ) {
    NVTX3_FUNC_RANGE();
    if (memory_type() == kayak::device_type::gpu) {
      herring::detail::predict<kayak::device_type::gpu>(
        obj(),
        out,
        in,
        num_class_,
        element_postproc_,
        row_postproc_,
        io_t(average_factor_),
        io_t(bias_),
        io_t(postproc_constant_),
        device_id,
        stream
      );
    } else {
      herring::detail::predict<kayak::device_type::cpu>(
        obj(),
        out,
        in,
        num_class_,
        element_postproc_,
        row_postproc_,
        io_t(average_factor_),
        io_t(bias_),
        io_t(postproc_constant_),
        device_id,
        stream
      );
    }
  }

  decision_forest() :
    tree_offsets_{},
    node_values_{},
    node_features_{},
    node_offsets_{},
    default_distant_{},
    node_outputs_{},
    categorical_sizes_{},
    categorical_storage_{},
    num_class_{},
    num_features_{},
    leaf_size_{} {}

  decision_forest(
    index_type num_class,
    index_type num_features,
    kayak::buffer<raw_index_t>&& tree_offsets,
    kayak::buffer<node_value_type>&& node_values,
    kayak::buffer<feature_index_type>&& node_features,
    kayak::buffer<offset_type>&& node_offsets,
    kayak::buffer<bool>&& default_distant,
    std::optional<kayak::buffer<output_type>>&& node_outputs,
    std::optional<kayak::buffer<raw_index_t>>&& categorical_sizes,
    std::optional<kayak::buffer<uint8_t>>&& categorical_storage,
    element_op element_postproc=element_op::disable,
    row_op row_postproc=row_op::disable,
    double average_factor=1.0,
    double bias=0.0,
    double postproc_constant=1.0,
    index_type leaf_size=index_type{1u}
  ) :
    tree_offsets_{std::move(tree_offsets)},
    node_values_{std::move(node_values)},
    node_features_{std::move(node_features)},
    node_offsets_{std::move(node_offsets)},
    default_distant_{std::move(default_distant)},
    node_outputs_{std::move(node_outputs)},
    categorical_sizes_{std::move(categorical_sizes)},
    categorical_storage_{std::move(categorical_storage)},
    num_class_{num_class},
    num_features_{num_features},
    element_postproc_{element_postproc},
    row_postproc_{row_postproc},
    average_factor_{average_factor},
    bias_{bias},
    postproc_constant_{postproc_constant},
    leaf_size_{leaf_size}
  {
    // TODO: Check for inconsistent memory type
  }

 private:
  kayak::buffer<raw_index_t> tree_offsets_;
  kayak::buffer<node_value_type> node_values_;
  kayak::buffer<feature_index_type> node_features_;
  kayak::buffer<offset_type> node_offsets_;
  kayak::buffer<bool> default_distant_;
  std::optional<kayak::buffer<output_type>> node_outputs_;
  std::optional<kayak::buffer<raw_index_t>> categorical_sizes_;
  std::optional<kayak::buffer<uint8_t>> categorical_storage_;

  // Metadata
  raw_index_t num_class_;
  raw_index_t num_features_;
  element_op element_postproc_;
  row_op row_postproc_;
  double average_factor_;
  double bias_;
  double postproc_constant_;
  raw_index_t leaf_size_;
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

using forest_model_variant = std::variant<
  forest_model<float, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<float, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<float, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<float, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<float, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<float, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<float, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<float, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<float, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<float, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<float, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<float, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<float, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<float, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<float, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<float, detail::preferred_tree_layout, true, true, true, true>,
  forest_model<double, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<double, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<double, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<double, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<double, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<double, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<double, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<double, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<double, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<double, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<double, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<double, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<double, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<double, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<double, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<double, detail::preferred_tree_layout, true, true, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<uint32_t, detail::preferred_tree_layout, true, true, true, true>
>;

inline auto get_forest_variant_index(
  std::size_t max_node_offset,
  std::size_t num_features,
  std::size_t max_num_categories,
  bool use_double_thresholds,
  bool use_double_output,
  bool use_integer_output
) {
  auto constexpr small_value = std::size_t{0};
  auto constexpr large_value = std::size_t{1};

  auto constexpr output_integer_bit = std::size_t{5};
  auto constexpr output_precision_bit = std::size_t{4};
  auto constexpr precision_bit = std::size_t{3};

  auto constexpr features_bit = std::size_t{2};
  auto constexpr max_few_features = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << features_bit), forest_model_variant
  >::feature_index_type>::max();
  auto constexpr max_many_features = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << features_bit), forest_model_variant
  >::feature_index_type>::max();

  auto constexpr tree_bit = std::size_t{1};
  auto constexpr max_small_trees = std::numeric_limits<typename std::variant_alternative_t<
    (small_value << tree_bit), forest_model_variant
  >::offset_type>::max();
  auto constexpr max_large_trees = std::numeric_limits<typename std::variant_alternative_t<
    (large_value << tree_bit), forest_model_variant
  >::offset_type>::max();

  auto constexpr category_bit = std::size_t{0};
  auto max_few_categories = std::variant_alternative_t<
    (small_value << category_bit), forest_model_variant
  >::category_set_type{}.size();
  auto constexpr max_many_categories = std::numeric_limits<raw_index_t>::max();

  if (num_features > max_many_features) {
    throw unusable_model_exception("Model contains too many features");
  }

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
    (std::size_t{use_integer_output} << output_integer_bit) +
    (std::size_t{use_double_output} << output_precision_bit) +
    (std::size_t{use_double_thresholds} << precision_bit) +
    (has_many_features << features_bit) +
    (has_large_trees << tree_bit) +
    (has_many_categories << category_bit)
  };
}
}
