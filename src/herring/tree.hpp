#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <herring/node.hpp>

namespace herring {
  /* A tree that can just return the stored value of nodes as its output */
  template <typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t>
  struct simple_tree {
    using node_type = simple_node<value_t, feature_index_t, offset_t, output_index_t>;
    using output_type = output_t;
    std::vector<node_type> nodes;
    std::vector<bool> default_distant;

    auto get_leaf_value(node_type const& node) const {
      if constexpr (std::is_same_v<value_t, output_t>) {
        return node.value.value;
      } else {
        static_assert(std::is_same_v<output_index_t, output_t>);
        return node.value.index;
      }
    }

    auto get_leaf_value(std::size_t node_index) const {
      return get_leaf_value(nodes[node_index]);
    }

    template<bool missing_values_in_row>
    auto evaluate_tree_node(std::size_t node_index, value_t const* row) const {
      auto const& node = nodes[node_index];
      if constexpr(missing_values_in_row) {
        auto feature_value = *(row + node.feature);
        auto present = !std::isnan(feature_value);
        auto result = offset_t{};
        if(present) {
          result = evaluate_node(node, feature_value);
        } else {
          result = 1 + (node.distant_offset - 1) * default_distant[node_index];
        }
        return result;
      } else {
        return evaluate_node(node, row);
      }
    }
  };


  /* A tree that must look up its output values in separate storage */
  template <typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t>
  struct lookup_tree {
    using node_type = simple_node<value_t, feature_index_t, offset_t, output_index_t>;
    using output_type = output_t;
    std::vector<node_type> nodes;
    std::vector<output_type> leaf_outputs;
    std::vector<offset_t> default_distant;

    // TODO (wphicks): return const& for vector leaves
    auto get_leaf_value(node_type const& node) const {
      return leaf_outputs[node.value.index];
    }
    auto get_leaf_value(std::size_t node_id) const {
      return leaf_outputs[nodes[node_id].value.index];
    }

    template<bool missing_values_in_row>
    auto evaluate_tree_node(std::size_t node_index, value_t const* row) const {
      auto const& node = nodes[node_index];
      if constexpr(missing_values_in_row) {
        auto feature_value = *(row + node.feature);
        auto present = !std::isnan(feature_value);
        if(present) {
          return evaluate_node(node, feature_value);
        } else {
          return 1 + (node.distant_offset - 1) * default_distant[node_index];
        }
      } else {
        return evaluate_node(node, row);
      }
    }
  };
}
