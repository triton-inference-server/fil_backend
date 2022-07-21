/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <herring/node.hpp>
#include "herring/type_helpers.hpp"

namespace herring {
  /* A tree that can just return the stored value of nodes as its output */
  template <typename value_t, typename feature_index_t, typename offset_t, typename output_index_t, typename output_t>
  struct simple_tree {
    using node_type = simple_node<value_t, feature_index_t, offset_t, output_index_t>;
    using output_type = output_t;
    std::vector<node_type> nodes;
    std::vector<bool> default_distant;
    std::vector<bool> categorical_node;
    bool has_categorical_nodes;

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

    template<bool missing_values_in_row, bool categorical_model, bool inclusive_threshold>
    auto evaluate_tree_node(std::size_t node_index, float const* row) const {
      auto result = offset_t{};
      if constexpr (categorical_model) {
        if (!has_categorical_nodes) {
          result = evaluate_tree_node_<missing_values_in_row, false, inclusive_threshold>(
            node_index, row
          );
        } else {
          result = evaluate_tree_node_<missing_values_in_row, true, inclusive_threshold>(
            node_index, row
          );
        }
      } else {
        result = evaluate_tree_node_<missing_values_in_row, false, inclusive_threshold>(
          node_index, row
        );
      }
      return result;
    };

   private:
    template<bool missing_values_in_row, bool categorical_tree, bool inclusive_threshold>
    auto evaluate_tree_node_(std::size_t node_index, float const* row) const {
      auto const& node = nodes[node_index];
      auto result = offset_t{};
      if constexpr(missing_values_in_row) {
        auto feature_value = *(row + node.feature);
        auto present = !std::isnan(feature_value);
        if (present) {
          if constexpr (categorical_tree) {
            if (!categorical_node[node_index]) {
              result = evaluate_node<false, inclusive_threshold>(node, feature_value);
            } else {
              result = evaluate_node<true, inclusive_threshold>(node, feature_value);
            }
          } else {
            result = evaluate_node<false, inclusive_threshold>(node, feature_value);
          }
        } else {
    // This narrowing conversion is guaranteed safe because distant_offset
    // cannot be 0
    // TODO(wphicks): Guarantee this with custom types
    // (https://github.com/triton-inference-server/fil_backend/issues/204)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
          result = 1 + (node.distant_offset - 1) * default_distant[node_index];
#pragma GCC diagnostic pop
        }
      } else {
        if constexpr (categorical_tree) {
          if (!categorical_node[node_index]) {
            result = evaluate_node<false, inclusive_threshold>(node, row);
          } else {
            result = evaluate_node<true, inclusive_threshold>(node, row);
          }
        } else {
          result = evaluate_node<false, inclusive_threshold>(node, row);
        }
      }
      return result;
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
    std::vector<bool> categorical_node;
    bool has_categorical_nodes;

    template <
      typename tree_output_type = output_t,
      std::enable_if_t<is_container_specialization<tree_output_type, std::vector>::value, bool> = true>
    auto const& get_leaf_value(node_type const& node) const {
      return leaf_outputs[node.value.index];
    }

    template <
      typename tree_output_type = output_t,
      std::enable_if_t<!is_container_specialization<tree_output_type, std::vector>::value, bool> = true>
    auto get_leaf_value(node_type const& node) const {
      return leaf_outputs[node.value.index];
    }

    auto get_leaf_value(std::size_t node_id) const {
      return leaf_outputs[nodes[node_id].value.index];
    }

    template<bool missing_values_in_row, bool categorical_model, bool inclusive_threshold>
    auto evaluate_tree_node(std::size_t node_index, float const* row) const {
      auto result = offset_t{};
      if constexpr (categorical_model) {
        if (!has_categorical_nodes) {
          result = evaluate_tree_node_<missing_values_in_row, false, inclusive_threshold>(
            node_index, row
          );
        } else {
          result = evaluate_tree_node_<missing_values_in_row, true, inclusive_threshold>(
            node_index, row
          );
        }
      } else {
        result = evaluate_tree_node_<missing_values_in_row, false, inclusive_threshold>(
          node_index, row
        );
      }
      return result;
    };

   private:
    template<bool missing_values_in_row, bool categorical_tree, bool inclusive_threshold>
    auto evaluate_tree_node_(std::size_t node_index, float const* row) const {
      auto const& node = nodes[node_index];
      auto result = offset_t{};
      if constexpr(missing_values_in_row) {
        auto feature_value = *(row + node.feature);
        auto present = !std::isnan(feature_value);
        if (present) {
          if constexpr (categorical_tree) {
            if (!categorical_node[node_index]) {
              result = evaluate_node<false, inclusive_threshold>(node, feature_value);
            } else {
              result = evaluate_node<true, inclusive_threshold>(node, feature_value);
            }
          } else {
            result = evaluate_node<false, inclusive_threshold>(node, feature_value);
          }
        } else {
    // This narrowing conversion is guaranteed safe because distant_offset
    // cannot be 0
    // TODO(wphicks): Guarantee this with custom types
    // (https://github.com/triton-inference-server/fil_backend/issues/204)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
          result = 1 + (node.distant_offset - 1) * default_distant[node_index];
#pragma GCC diagnostic pop
        }
      } else {
        if constexpr (categorical_tree) {
          if (!categorical_node[node_index]) {
            result = evaluate_node<false, inclusive_threshold>(node, row);
          } else {
            result = evaluate_node<true, inclusive_threshold>(node, row);
          }
        } else {
          result = evaluate_node<false, inclusive_threshold>(node, row);
        }
      }
    return result;
    }
  };
}
