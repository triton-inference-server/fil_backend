#pragma once
#include <nvtx3/nvtx3.hpp>
#include <math.h>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/bitset.hpp>
#include <kayak/flat_array.hpp>
#include <kayak/gpu_support.hpp>
#include <herring3/node_value.hpp>
#include <iostream>
#include <kayak/tree.hpp>
#include <kayak/tree_layout.hpp>

namespace herring {

using kayak::raw_index_t;

template<typename storage_t>
struct node_metadata_storage {
  using storage_type = storage_t;
  HOST DEVICE node_metadata_storage() : data{} {}
  HOST DEVICE node_metadata_storage(
    bool is_leaf_node,
    bool default_to_distant_child,
    bool is_categorical_node,
    storage_type feature = storage_type{}
  ) : data(
    (is_leaf_node << LEAF_BIT) +
    (default_to_distant_child << DEFAULT_DISTANT_BIT) +
    (is_categorical_node << CATEGORICAL_BIT) +
    (feature & FEATURE_MASK)
  ) {
  }
  HOST DEVICE auto feature_index() const {
    return storage_type(data & FEATURE_MASK);
  }
  HOST DEVICE auto is_leaf() const {
    return bool(data & LEAF_MASK);
  }
  HOST DEVICE auto default_distant() const {
    return bool(data & DEFAULT_DISTANT_MASK);
  }
  HOST DEVICE auto is_categorical() const {
    return bool(data & CATEGORICAL_MASK);
  }

 private:
  storage_type data;
  auto constexpr static const LEAF_BIT = storage_type(sizeof(storage_type) * 8 - 1);
  auto constexpr static const LEAF_MASK = storage_type(1 << LEAF_BIT);
  auto constexpr static const DEFAULT_DISTANT_BIT = storage_type(LEAF_BIT - 1);
  auto constexpr static const DEFAULT_DISTANT_MASK = storage_type(1 << DEFAULT_DISTANT_BIT);
  auto constexpr static const CATEGORICAL_BIT = storage_type(DEFAULT_DISTANT_BIT - 1);
  auto constexpr static const CATEGORICAL_MASK = storage_type(1 << CATEGORICAL_BIT);
  auto constexpr static const FEATURE_MASK = storage_type(
    ~(LEAF_MASK | DEFAULT_DISTANT_MASK | CATEGORICAL_MASK)
  );
};

template<kayak::tree_layout layout, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t, bool categorical_lookup>
struct forest {
  auto constexpr static const bounds_check = kayak::DEBUG_ENABLED && !kayak::GPU_ENABLED;
  using index_type = kayak::detail::index_type<bounds_check>;
  using feature_index_type = feature_index_t;
  using category_set_type = std::conditional_t<
    categorical_lookup,
    kayak::bitset<uint8_t>,
    kayak::bitset<output_index_t>
  >;
  using node_value_type = node_value<value_t, output_index_t>;
  using offset_type = offset_t;
  using output_index_type = output_index_t;
  using output_type = output_t;
  using value_type = value_t;
  using metadata_type = node_metadata_storage<feature_index_type>;

  forest()
    : node_count_{}, values_{nullptr}, features_{nullptr},
    distant_offsets_{nullptr}, metadata_{nullptr}, tree_count_{},
    tree_offsets_{nullptr}, output_size_{}, outputs_{nullptr},
    categorical_sizes_{nullptr}, categorical_storage_{nullptr} { }

  forest(
    index_type node_count,
    node_value_type* node_values,
    feature_index_t* node_features,
    offset_type* distant_child_offsets,
    metadata_type* metadata,
    index_type tree_count,
    raw_index_t* tree_offsets,
    index_type output_size = raw_index_t{1},
    output_t* outputs = nullptr,
    raw_index_t* categorical_sizes = nullptr,
    uint8_t* categorical_storage = nullptr
  ) : node_count_{node_count}, values_{node_values}, features_{node_features},
    distant_offsets_{distant_child_offsets}, metadata_{metadata}, tree_count_{tree_count},
    tree_offsets_{tree_offsets}, output_size_{output_size}, outputs_{outputs},
    categorical_sizes_{categorical_sizes}, categorical_storage_{categorical_storage} { }

  raw_index_t node_count_;
  node_value_type* values_;
  feature_index_t* features_;
  offset_type* distant_offsets_;
  metadata_type* metadata_;

  raw_index_t tree_count_;
  raw_index_t* tree_offsets_;

  raw_index_t output_size_;
  // Optional data (may be null)
  output_t* outputs_;
  raw_index_t* categorical_sizes_;
  uint8_t* categorical_storage_;
};

}
