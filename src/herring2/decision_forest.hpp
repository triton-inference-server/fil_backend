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
  using index_type = typename forest_type::index_type;
  using feature_index_type = typename forest_type::feature_index_type;
  using offset_type = typename forest_type::offset_type;
  using output_index_type = typename forest_type::output_index_type;
  using output_type = typename forest_type::output_type;
  using node_value_type = typename forest_type::node_value_type;
  using category_set_type = typename forest_type::category_set_type;

  void set_offset(index_type tree_index, index_type node_index, offset_type value) {
    // TODO(wphicks)
  }
  void set_value(index_type tree_index, index_type node_index, typename node_value_type::value_type value) {
    // TODO(wphicks)
  }
  void set_value(index_type tree_index, index_type node_index, typename node_value_type::output_index_type value) {
    // TODO(wphicks)
  }
  void set_feature(index_type tree_index, index_type node_index, feature_index_type value) {
    // TODO(wphicks)
  }
  void set_default_distant(index_type tree_index, index_type node_index, bool value) {
    // TODO(wphicks)
  }
  void set_output(index_type tree_index, index_type node_index, output_type value) {
    // TODO(wphicks)
  }
  template<kayak::array_encoding output_layout>
  void set_output(index_type tree_index, index_type node_index, kayak::flat_array<output_layout, output_type> value) {
    // TODO(wphicks)
  }
  void set_categories(index_type tree_index, index_type node_index, kayak::flat_array<kayak::array_encoding::dense, typename category_set_type::index_type> value) {
    // TODO(wphicks)
  }


  auto num_features() const { return num_features_; }
  auto leaf_size() const { return leaf_size_; }

  auto obj() const {
    auto node_output_ptr = static_cast<output_type*>(nullptr);
    if (node_outputs_) {
      node_output_ptr = node_outputs_->data();
    }
    auto categorical_sizes__ptr = static_cast<raw_index_t*>(nullptr);
    if (categorical_sizes_) {
      categorical_sizes__ptr = categorical_sizes_->data();
    }
    auto categorical_storage__ptr = static_cast<uint8_t*>(nullptr);
    if (categorical_storage_) {
      categorical_storage__ptr = categorical_storage_->data();
    }
    // TODO (wphicks)
    /* return forest{
      node_offsets_,
      node_values_.data(),
      node_features_.data(),
      node_offsets_.data(),
      default_distant_.data(),
      node_offsets_.size(),
      leaf_size_,
      node_output_ptr,
      categorical_sizes__ptr,
      categorical_storage__ptr
    }; */
  }

  decision_forest() :
    tree_offsets_{},
    node_values_{},
    node_features_{},
    node_offsets_{},
    default_distant_{},
    node_outputs_{},
    categorical_sizes_{},
    categorical_storage_{} {}

  template<typename iter>
  decision_forest(
    index_type num_class,
    index_type num_features,
    iter tree_sizes_begin,
    iter tree_sizes_end,
    index_type align_bytes=raw_index_t{},
    kayak::device_type mem_type=kayak::device_type::cpu,
    int device=0,
    kayak::cuda_stream stream=kayak::cuda_stream{}
  ) :
      tree_offsets_{[this, &tree_sizes_begin, &tree_sizes_end, align_bytes, mem_type, device, &stream]() {
      auto offsets = std::vector<raw_index_t>{raw_index_t{}};
      offsets.reserve(std::distance(tree_sizes_begin, tree_sizes_end) + 1);
      auto alignment = std::lcm(align_unit, align_bytes);
      std::transform(
        tree_sizes_begin,
        tree_sizes_end,
        std::back_inserter(offsets),
        [this, align_bytes, mem_type, device, &stream, &offsets](auto&& current) {
          auto result = raw_index_t{};
          auto const& cumulative = offsets.back();
          if (align_bytes == 0) {
            result = cumulative + current;
          } else {
            result = cumulative + current + (alignment - current % alignment);
          }
        }
      );
      offsets.pop_back();
      return kayak::buffer<raw_index_t>{
        std::begin(offsets), std::end(offsets), mem_type, device, stream
      };
    }()},
    node_values_{tree_offsets_.size(), mem_type, device, stream},
    node_features_{tree_offsets_.size(), mem_type, device, stream},
    default_distant_{tree_offsets_.size(), mem_type, device, stream},
    node_outputs_{},
    categorical_sizes_{},
    categorical_storage_{} {
  }

 private:
  // Data
  kayak::buffer<raw_index_t> tree_offsets_;
  kayak::buffer<node_value_type> node_values_;
  kayak::buffer<feature_index_type> node_features_;
  kayak::buffer<offset_type> node_offsets_;
  kayak::buffer<bool> default_distant_;
  std::optional<kayak::buffer<output_type>> node_outputs_;
  std::optional<kayak::buffer<raw_index_t>> categorical_sizes_;
  std::optional<kayak::buffer<uint8_t>> categorical_storage_;

  auto constexpr static const align_unit = std::lcm(
    sizeof(raw_index_t),
    std::lcm(
      sizeof(node_value_type),
      std::lcm(
        sizeof(feature_index_type),
        sizeof(offset_type)
      )
    )
  );

  // Metadata
  raw_index_t num_class_;
  raw_index_t num_features_;
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

namespace detail {
auto constexpr static const preferred_tree_layout = kayak::tree_layout::depth_first;
}

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
  forest_model<uint64_t, detail::preferred_tree_layout, false, false, false, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, false, false, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, false, true, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, false, true, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, true, false, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, true, false, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, true, true, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, false, true, true, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, false, false, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, false, false, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, false, true, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, false, true, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, true, false, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, true, false, true>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, true, true, false>,
  forest_model<uint64_t, detail::preferred_tree_layout, true, true, true, true>
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
