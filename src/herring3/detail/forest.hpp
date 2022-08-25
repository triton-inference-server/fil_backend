#pragma once
#include <stddef.h>
#include <kayak/gpu_support.hpp>
#include <herring3/detail/node.hpp>

namespace herring {

/** A collection of trees which together form a forest model
 */
template <kayak::tree_layout layout_v, typename threshold_t, typename index_t, typename metadata_storage_t, typename offset_t>
struct forest {
  using node_type = node<layout_v, threshold_t, index_t, metadata_storage_t, offset_t>;
  using io_type = threshold_t;

  HOST DEVICE forest(node_type* forest_nodes, size_t* forest_root_indexes, size_t num_trees) :
    nodes_{forest_nodes}, root_node_indexes_{forest_root_indexes}, num_trees_{num_trees} {}

  /** Return pointer to the root node of the indicated tree */
  HOST DEVICE auto* get_tree_root(size_t tree_index) const {
    return nodes_ + root_node_indexes_[tree_index];
  }

  /** Return the number of trees in this forest */
  HOST DEVICE auto tree_count() const {
    return num_trees_;
  }
 private:
  node_type* nodes_;
  size_t* root_node_indexes_;
  size_t num_trees_;
};

}
