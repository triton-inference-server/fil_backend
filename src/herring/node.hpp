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
#include <bitset>
#include <cstdint>

namespace herring {
  /* Summary of Types
   * ----------------
   *  value_t (float or double): The value used for testing a node condition or
   *    for providing the output of leaves.
   *  feature_index_t (std::uint16_t or std::uint32_t): Index indicating which
   *    feature this conditional applies to
   *  offset_t (std::uint16_t or std::uint32_t): Offset between this node and
   *    its distant child. For small trees, using a smaller type can reduce the
   *    padded size of the node to as few as 8 bytes.
   *  output_index_t (typically std::uint32_t): If leaf output values cannot be stored
   *    in the same memory as test condition values, this index provides a
   *    lookup location for output values stored in the tree.
   */
  template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t>
  struct simple_node {
    using value_type = value_t;  // float or double
    using index_type = feature_index_t;
    using offset_type = offset_t;
    using output_index_type = output_index_t;
    // Cannot use std::variant here because it takes up 4 additional bytes when
    // value_type is float
    union value_or_index {
      value_type value;
      output_index_type index;
      std::bitset<std::max(sizeof(value_type), sizeof(output_index_type))> categories;
    };
    value_or_index value;  // 4 bytes for float
    offset_type distant_offset;  // 2 bytes for depth < 16 or small trees; 4 otherwise
    index_type feature; // 1-4 bytes, depending on number of features
  };

  template<bool categorical, bool inclusive_threshold, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t>
  auto evaluate_node(simple_node<value_t, feature_index_t, offset_t, output_index_t> const& node, float feature_value) {
    auto condition = false;
    if constexpr (categorical) {
      if constexpr (inclusive_threshold) {
        if (feature_value > 0 && feature_value < node.value.categories.size()) {
          condition = node.value.categories[feature_value];
        }
      } else {
        if (feature_value > 0 && feature_value < node.value.categories.size()) {
          condition = !node.value.categories[feature_value];
        } else {
          condition = true;
        }
      }
    } else {
      if constexpr (inclusive_threshold) {
        condition = (feature_value <= node.value.value);
      } else {
        condition = (feature_value < node.value.value);
      }
    }

    // This narrowing conversion is guaranteed safe because distant_offset
    // cannot be 0
    // TODO(wphicks): Guarantee this with custom types
    // (https://github.com/triton-inference-server/fil_backend/issues/204)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return offset_t{1 + condition * (node.distant_offset - 1)};
#pragma GCC diagnostic pop
  }

  template<bool categorical, bool inclusive_threshold, typename value_t, typename feature_index_t, typename offset_t, typename output_index_t>
  auto evaluate_node(simple_node<value_t, feature_index_t, offset_t, output_index_t> const& node, float const* row) {
    auto feature_value = *(row + node.feature);
    return evaluate_node<categorical, inclusive_threshold>(node, feature_value);
  }
}
