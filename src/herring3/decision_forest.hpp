#pragma once
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <herring3/constants.hpp>
#include <herring3/postproc_ops.hpp>
#include <herring3/detail/device_initialization.hpp>
#include <herring3/detail/infer.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <herring3/exceptions.hpp>
#include <herring3/detail/forest.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/exceptions.hpp>
#include <kayak/tree_layout.hpp>
#include <limits>
#include <optional>
#include <variant>

namespace herring {

template <kayak::tree_layout layout_v, typename threshold_t, typename index_t, typename metadata_storage_t, typename offset_t>
struct decision_forest {

  auto constexpr static const layout = layout_v;
  using forest_type = forest<
    layout,
    threshold_t,
    index_t,
    metadata_storage_t,
    offset_t
  >;
  using node_type = typename forest_type::node_type;
  using io_type = typename forest_type::io_type;
  using threshold_type = threshold_t;
  using postprocessor_type = postprocessor<io_type>;
  using categorical_storage_type = typename node_type::index_type;

  decision_forest() :
    nodes_{},
    root_node_indexes_{},
    vector_output_{},
    categorical_storage_{},
    num_feature_{},
    num_class_{},
    leaf_size_{},
    has_categorical_nodes_{false},
    row_postproc_{},
    elem_postproc_{},
    average_factor_{},
    bias_{},
    postproc_constant_{} {}

  decision_forest(
    kayak::buffer<node_type>&& nodes,
    kayak::buffer<size_t>&& root_node_indexes,
    size_t num_feature,
    size_t num_class=size_t{2},
    bool has_categorical_nodes = false,
    std::optional<kayak::buffer<io_type>>&& vector_output=std::nullopt,
    std::optional<kayak::buffer<typename node_type::index_type>>&& categorical_storage=std::nullopt,
    size_t leaf_size=size_t{1},
    row_op row_postproc=row_op::disable,
    element_op elem_postproc=element_op::disable,
    io_type average_factor=io_type{1},
    io_type bias=io_type{0},
    io_type postproc_constant=io_type{1}
  ) :
    nodes_{nodes},
    root_node_indexes_{root_node_indexes},
    vector_output_{vector_output},
    categorical_storage_{categorical_storage},
    num_feature_{num_feature},
    num_class_{num_class},
    leaf_size_{leaf_size},
    has_categorical_nodes_{has_categorical_nodes},
    row_postproc_{row_postproc},
    elem_postproc_{elem_postproc},
    average_factor_{average_factor},
    bias_{bias},
    postproc_constant_{postproc_constant}
  {
    if (nodes.memory_type() != root_node_indexes.memory_type()) {
      throw kayak::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on either host or device"
      );
    }
    if (nodes.device_index() != root_node_indexes.device_index()) {
      throw kayak::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on same device"
      );
    }
    detail::initialize_device<forest_type>(nodes.device());
  }

  auto num_feature() const { return num_feature_; }
  auto num_outputs() const { return num_class_; }

  auto memory_type() {
    return nodes_.memory_type();
  }
  auto device_index() {
    return nodes_.device_index();
  }

  void predict(
    kayak::buffer<typename forest_type::io_type>& output,
    kayak::buffer<typename forest_type::io_type> const& input,
    kayak::cuda_stream stream = kayak::cuda_stream{},
    std::optional<std::size_t> specified_rows_per_block_iter=std::nullopt
  ) {
    if (output.memory_type() != memory_type() || input.memory_type() != memory_type()) {
      throw kayak::wrong_device_type{
        "Tried to use host I/O data with model on device or vice versa"
      };
    }
    if (output.device_index() != device_index() || input.device_index() != device_index()) {
      throw kayak::wrong_device{
        "I/O data on different device than model"
      };
    }
    auto* vector_output_data = (
      vector_output_.has_value() ? vector_output_->data() : static_cast<io_type*>(nullptr)
    );
    auto* categorical_storage_data = (
      categorical_storage_.has_value() ? categorical_storage_->data() : static_cast<categorical_storage_type*>(nullptr)
    );
    switch(nodes_.device().index()) {
      case 0:
        herring::detail::infer(
          obj(),
          get_postprocessor(),
          output.data(),
          input.data(),
          input.size() / num_feature_,
          num_feature_,
          num_class_,
          has_categorical_nodes_,
          vector_output_data,
          categorical_storage_data,
          specified_rows_per_block_iter,
          std::get<0>(nodes_.device()),
          stream
        );
        break;
      case 1:
        herring::detail::infer(
          obj(),
          get_postprocessor(),
          output.data(),
          input.data(),
          input.size() / num_feature_,
          num_feature_,
          num_class_,
          has_categorical_nodes_,
          vector_output_data,
          categorical_storage_data,
          specified_rows_per_block_iter,
          std::get<1>(nodes_.device()),
          stream
        );
        break;
    }
  }

 private:
  /** The nodes for all trees in the forest */
  kayak::buffer<node_type> nodes_;
  /** The index of the root node for each tree in the forest */
  kayak::buffer<size_t> root_node_indexes_;
  /** Buffer of outputs for all leaves in vector-leaf models */
  std::optional<kayak::buffer<io_type>> vector_output_;
  /** Buffer of outputs for all leaves in vector-leaf models */
  std::optional<kayak::buffer<categorical_storage_type>> categorical_storage_;

  // Metadata
  size_t num_feature_;
  size_t num_class_;
  size_t leaf_size_;
  bool has_categorical_nodes_ = false;
  // Postprocessing constants
  row_op row_postproc_;
  element_op elem_postproc_;
  io_type average_factor_;
  io_type bias_;
  io_type postproc_constant_;

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

  auto leaf_size() const { return leaf_size_; }
};

namespace detail {
template<
  kayak::tree_layout layout,
  bool double_precision,
  bool large_trees
>
using preset_decision_forest = decision_forest<
  layout,
  std::conditional_t<double_precision, double, float>,
  std::conditional_t<double_precision, uint64_t, uint32_t>,
  std::conditional_t<large_trees, uint32_t, uint16_t>,
  std::conditional_t<large_trees, uint32_t, uint16_t>
>;
}

using decision_forest_variant = std::variant<
  detail::preset_decision_forest<preferred_tree_layout, false, false>,
  detail::preset_decision_forest<preferred_tree_layout, false, true>,
  detail::preset_decision_forest<preferred_tree_layout, true, false>,
  detail::preset_decision_forest<preferred_tree_layout, true, true>
>;

inline auto get_forest_variant_index(
  bool use_double_thresholds,
  std::size_t max_node_offset,
  std::size_t num_features,
  std::size_t num_categorical_nodes = std::size_t{},
  std::size_t max_num_categories = std::size_t{},
  std::size_t num_vector_leaves = std::size_t{}
) {
  auto max_local_categories = sizeof(std::uint32_t) * 8;
  // If the index required for pointing to categorical storage bins or vector
  // leaf output exceeds what we can store in a uint32_t, uint64_t will be used
  //
  // TODO(wphicks): We are overestimating categorical storage required here
  auto double_indexes_required = (
    max_num_categories > max_local_categories
    && (
      (
        kayak::ceildiv(max_num_categories, max_local_categories) + 1
        * num_categorical_nodes
      ) > std::numeric_limits<std::uint32_t>::max()
    )
  ) || num_vector_leaves > std::numeric_limits<std::uint32_t>::max();

  auto double_precision = use_double_thresholds || double_indexes_required;

  auto large_trees = (
    num_features > (
      std::numeric_limits<std::uint16_t>::max() >> reserved_node_metadata_bits
    ) || max_node_offset > std::numeric_limits<std::uint16_t>::max()
  );

  return (
    (std::size_t{double_precision} << std::size_t{1})
    + std::size_t{large_trees}
  );
}
}
