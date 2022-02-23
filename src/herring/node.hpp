#pragma once
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
    };
    value_or_index value;  // 4 bytes for float
    offset_type distant_offset;  // 2 bytes for depth < 16 or small trees; 4 otherwise
    index_type feature; // 1-4 bytes, depending on number of features
  };

  template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t>
  auto evaluate_node(simple_node<value_t, feature_index_t, offset_t, output_index_t> const& node, value_t const* row) {
    // This narrowing conversion is guaranteed safe because distant_offset
    // cannot be 0
    // TODO(wphicks): Guarantee this with custom types
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return offset_t{1 + (*(row + node.feature) < node.value.value) * (node.distant_offset - 1)};
#pragma GCC diagnostic pop
  }

  template<typename value_t, typename feature_index_t, typename offset_t, typename output_index_t>
  auto evaluate_node(simple_node<value_t, feature_index_t, offset_t, output_index_t> const& node, value_t feature_value) {
    // This narrowing conversion is guaranteed safe because distant_offset
    // cannot be 0
    // TODO(wphicks): Guarantee this with custom types
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return offset_t{1 + (feature_value < node.value.value) * (node.distant_offset - 1)};
#pragma GCC diagnostic pop
  }
}
