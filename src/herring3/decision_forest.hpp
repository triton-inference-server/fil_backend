#pragma once
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <herring3/constants.hpp>
#include <herring3/postprocessor.hpp>
#include <herring3/exceptions.hpp>
#include <herring3/detail/forest.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/tree_layout.hpp>
#include <limits>
#include <optional>
#include <variant>

namespace herring {

template <kayak::tree_layout layout_v, typename threshold_t, typename index_t, typename metadata_storage_t, typename offset_t, typename leaf_output_t>
struct decision_forest {

  auto constexpr static const layout = layout_v;
  using forest_type = forest<
    layout,
    threshold_t,
    index_t,
    metadata_storage_t,
    offset_t,
    leaf_output_t
  >;
  using node_type = typename forest_type::node_type;
  using io_type = typename forest_type::io_type;
  using threshold_type = threshold_t;
  using leaf_output_type = typename forest_type::leaf_output_type;
  using postprocessor_type = postprocessor<leaf_output_type, io_type>;

  auto num_class() const { return num_class_; }
  auto leaf_size() const { return leaf_size_; }

  auto obj() const {
    return forest_type{
      nodes_.data(),
      root_node_indexes_.data(),
      root_node_indexes_.size()
    };
  }

  auto get_postprocessor() const {
    return postprocessor_type {
      row_postproc_,
      elem_postproc_,
      average_factor_,
      bias_,
      postproc_constant_
    };
  }

  decision_forest() :
    nodes_{},
    root_node_indexes_{},
    num_class_{},
    leaf_size_{},
    row_postproc_{},
    elem_postproc_{},
    average_factor_{},
    bias_{},
    postproc_constant_{} {}

  decision_forest(
    kayak::buffer<node_type>&& nodes,
    kayak::buffer<size_t>&& root_node_indexes,
    size_t num_class=size_t{2},
    size_t leaf_size=size_t{1},
    row_op row_postproc=row_op::disable,
    element_op elem_postproc=element_op::disable,
    io_type average_factor=io_type{1},
    io_type bias=io_type{0},
    io_type postproc_constant=io_type{1}
  ) :
    nodes_{nodes},
    root_node_indexes_{root_node_indexes},
    num_class_{num_class},
    leaf_size_{leaf_size},
    row_postproc_{row_postproc},
    elem_postproc_{elem_postproc},
    average_factor_{average_factor},
    bias_{bias},
    postproc_constant_{postproc_constant}
  {
    // TODO: Check for inconsistent memory type
  }

 private:
  /** The nodes for all trees in the forest */
  kayak::buffer<node_type> nodes_;
  /** The index of the root node for each tree in the forest */
  kayak::buffer<size_t> root_node_indexes_;

  // Metadata
  size_t num_class_;
  size_t leaf_size_;
  // Postprocessing constants
  row_op row_postproc_;
  element_op elem_postproc_;
  io_type average_factor_;
  io_type bias_;
  io_type postproc_constant_;
};

template<
  typename leaf_output_t,
  kayak::tree_layout layout,
  bool double_precision,
  bool many_features,
  bool large_trees
>
using forest_model = decision_forest<
  layout,
  std::conditional_t<double_precision, double, float>,
  uint32_t,
  std::conditional_t<many_features, uint32_t, uint16_t>,
  std::conditional_t<large_trees, uint32_t, uint16_t>,
  leaf_output_t
>;

using forest_model_variant = std::variant<
  forest_model<float, preferred_tree_layout, false, false, false>
>;

inline auto get_forest_variant_index(
  std::size_t max_node_offset,
  std::size_t num_features,
  std::size_t max_num_categories,
  bool use_double_thresholds,
  bool use_double_output,
  bool use_integer_output
) {
  return std::size_t{};
}
}
