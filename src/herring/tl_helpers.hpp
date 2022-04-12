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

#include <exception>
#include <iostream>
#include <stack>
#include <string>
#include <variant>

#include <treelite/tree.h>

#include <herring/model.hpp>
#include <herring/output_ops.hpp>

namespace herring {

struct unconvertible_model_exception : std::exception {
  unconvertible_model_exception () : msg_{"Model could not be converted"}
  {
  }

  unconvertible_model_exception (std::string msg) : msg_{msg}
  {
  }

  unconvertible_model_exception (char const* msg) : msg_{msg}
  {
  }

  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

// TODO(wphicks): Currently, the model use_inclusive_threshold parameter is
// changed as a side-effect of this function. This is messy and confusing, and
// it should be fixed in a later refactor
// https://github.com/triton-inference-server/fil_backend/issues/205
template<typename tree_t, typename tl_threshold_t, typename tl_output_t>
auto convert_tree(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, bool& use_inclusive_threshold) {
  auto result = tree_t{};
  result.nodes.reserve(tl_tree.num_nodes);
  result.default_distant.reserve(tl_tree.num_nodes);

  // TL node id for current node
  auto cur_node_id = int{};
  // TL node id for hot child of current node
  auto hot_child = int{};
  // TL node id for distant child of current node
  auto distant_child = int{};

  // Stack of TL node ids for DFS
  auto node_stack = std::stack<int, std::vector<int>>{};
  // Keep track of final location of parent in nodes vector for each node currently in stack
  auto parent_stack = std::stack<std::size_t, std::vector<std::size_t>>{};
  // TODO(wphicks): Just store a reference to the parent directly rather than
  // an index
  // https://github.com/triton-inference-server/fil_backend/issues/205

  // Start at TL node id 0
  node_stack.push(cur_node_id);

  // Depth-first traversal, with hot child always searched first
  while (not node_stack.empty()) {
    cur_node_id = node_stack.top();
    node_stack.pop();

    // Tell parent where its distant child landed
    if (not parent_stack.empty()) {
      auto parent_index = parent_stack.top();
      parent_stack.pop();
      // Don't care if it overwrites; we always visit distant child last
      result.nodes[parent_index].distant_offset = result.nodes.size() - parent_index;
    }

    result.nodes.emplace_back();
    auto& cur_node = result.nodes.back();

    if (tl_tree.IsLeaf(cur_node_id)) {
      cur_node.distant_offset = typename tree_t::node_type::offset_type{};  // 0 offset means no child

      if constexpr (std::is_same_v<typename tree_t::output_type, decltype(tl_tree.LeafVector(0))>) {
        if (tl_tree.HasLeafVector(cur_node_id)) {
          cur_node.value.index = result.leaf_outputs.size();
          result.leaf_outputs.push_back(tl_tree.LeafVector(cur_node_id));
        } else {
          throw unconvertible_model_exception{"Leaf vector expected"};
        }
      } else {
        if constexpr (std::is_same_v<typename tree_t::node_type::value_type, typename tree_t::output_type>) {
          // Threshold and output values are the same type; store in same union
          // attribute
          cur_node.value.value = tl_tree.LeafValue(cur_node_id);
        } else if constexpr (std::is_same_v<typename tree_t::output_type, typename tree_t::node_type::output_index_type>) {
          // Threshold and output value types are different, but output value
          // type happens to be the same as index type; use index union attribute
          cur_node.value.index = tl_tree.LeafValue(cur_node_id);
        } else {
          // Threshold and output value types are different; output value must
          // be stored independently
          cur_node.value.index = result.leaf_outputs.size();
          result.leaf_outputs.push_back(tl_tree.LeafValue(cur_node_id));
        }
      }
      result.default_distant.push_back(false);
    } else {
      cur_node.feature = tl_tree.SplitIndex(cur_node_id);

      auto left_id = tl_tree.LeftChild(cur_node_id);
      auto right_id = tl_tree.RightChild(cur_node_id);
      auto default_child = tl_tree.DefaultChild(cur_node_id);
      auto tl_operator = tl_tree.ComparisonOp(cur_node_id);
      auto tl_split = tl_tree.SplitType(cur_node_id);
      auto categorical = (tl_split == treelite::SplitFeatureType::kCategorical);

      // Hot child is always less-than or in-category condition
      if (!categorical) {
        cur_node.value.value = tl_tree.Threshold(cur_node_id);
        auto inclusive_threshold_node = (
          tl_operator == treelite::Operator::kLE || tl_operator == treelite::Operator::kGE
        );
        if (!inclusive_threshold_node && use_inclusive_threshold) {
          throw unconvertible_model_exception{"Inconsistent use of inclusive threshold"};
        } else {
          use_inclusive_threshold = inclusive_threshold_node;
        }
        if (
            tl_operator == treelite::Operator::kLT || tl_operator == treelite::Operator::kLE) {
          hot_child = right_id;
          distant_child = left_id;
        } else if (
            tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kGE) {
          hot_child = left_id;
          distant_child = right_id;
        } else {
          throw unconvertible_model_exception{"Unsupported comparison operator"};
        }
      } else {
        if (tl_tree.CategoriesListRightChild()) {
          hot_child = right_id;
          distant_child = left_id;
        } else {
          hot_child = left_id;
          distant_child = right_id;
        }
        auto tl_categories = tl_tree.MatchingCategories();
        auto bit_representation = std::transform_reduce(
          std::begin(tl_categories),
          std::end(tl_categories),
          std::size_t{},
          std::plus<>(),
          [](auto&& category) {
            if (category > tree_t::node_type::category_set_type::size()) {
              throw unconvertible_model_exception{
                "Too many categories for size of category storage"
              };
            }
            return std::size_t{1} << category;
          }
        );
        cur_node.value.categories = typename tree_t::node_type::category_set_type{bit_representation};
      }

      result.default_distant.push_back(distant_child == default_child);

      node_stack.push(distant_child);
      node_stack.push(hot_child);
      parent_stack.push(result.nodes.size() - 1);
      parent_stack.push(result.nodes.size() - 1);

    } // End of handling for non-leaf nodes
  } // node_stack is empty; DFS is done

  return result;
} // convert_tree

using tl_dispatched_model = std::variant<
  // value_type, feature_index_type, offset_type, output_index_type, output_type
  simple_model<float, std::uint16_t, std::uint16_t, std::uint32_t, std::uint32_t>,
  simple_model<float, std::uint16_t, std::uint16_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<float, std::uint16_t, std::uint16_t, std::uint32_t, float>,
  simple_model<float, std::uint16_t, std::uint16_t, std::uint32_t, std::vector<float>>,

  simple_model<float, std::uint16_t, std::uint32_t, std::uint32_t, std::uint32_t>,
  simple_model<float, std::uint16_t, std::uint32_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<float, std::uint16_t, std::uint32_t, std::uint32_t, float>,
  simple_model<float, std::uint16_t, std::uint32_t, std::uint32_t, std::vector<float>>,

  simple_model<float, std::uint32_t, std::uint16_t, std::uint32_t, std::uint32_t>,
  simple_model<float, std::uint32_t, std::uint16_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<float, std::uint32_t, std::uint16_t, std::uint32_t, float>,
  simple_model<float, std::uint32_t, std::uint16_t, std::uint32_t, std::vector<float>>,

  simple_model<float, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
  simple_model<float, std::uint32_t, std::uint32_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<float, std::uint32_t, std::uint32_t, std::uint32_t, float>,
  simple_model<float, std::uint32_t, std::uint32_t, std::uint32_t, std::vector<float>>,

  simple_model<double, std::uint16_t, std::uint16_t, std::uint32_t, std::uint32_t>,
  simple_model<double, std::uint16_t, std::uint16_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<double, std::uint16_t, std::uint16_t, std::uint32_t, double>,
  simple_model<double, std::uint16_t, std::uint16_t, std::uint32_t, std::vector<double>>,

  simple_model<double, std::uint16_t, std::uint32_t, std::uint32_t, std::uint32_t>,
  simple_model<double, std::uint16_t, std::uint32_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<double, std::uint16_t, std::uint32_t, std::uint32_t, double>,
  simple_model<double, std::uint16_t, std::uint32_t, std::uint32_t, std::vector<double>>,

  simple_model<double, std::uint32_t, std::uint16_t, std::uint32_t, std::uint32_t>,
  simple_model<double, std::uint32_t, std::uint16_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<double, std::uint32_t, std::uint16_t, std::uint32_t, double>,
  simple_model<double, std::uint32_t, std::uint16_t, std::uint32_t, std::vector<double>>,

  simple_model<double, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
  simple_model<double, std::uint32_t, std::uint32_t, std::uint32_t, std::vector<std::uint32_t>>,
  simple_model<double, std::uint32_t, std::uint32_t, std::uint32_t, double>,
  simple_model<double, std::uint32_t, std::uint32_t, std::uint32_t, std::vector<double>>
>;

template<std::size_t model_variant_index, typename tl_threshold_t, typename tl_output_t>
auto convert_dispatched_model(treelite::ModelImpl<tl_threshold_t, tl_output_t> const& tl_model) {
  using model_type = std::variant_alternative_t<model_variant_index, tl_dispatched_model>;
  auto result = model_type{};
  result.use_inclusive_threshold = false;

  result.trees.reserve(tl_model.trees.size());
  std::transform(
    std::begin(tl_model.trees),
    std::end(tl_model.trees),
    std::back_inserter(result.trees),
    [&result](auto&& tl_tree) { return convert_tree<typename model_type::tree_type>(tl_tree, result.use_inclusive_threshold); }
  );

  result.num_class = tl_model.task_param.num_class;
  result.num_feature = tl_model.num_feature;
  if (tl_model.average_tree_output) {
    if (tl_model.task_type == treelite::TaskType::kMultiClfGrovePerClass) {
      result.average_factor = tl_model.trees.size() / result.num_class;
    } else {
      result.average_factor = tl_model.trees.size();
    }
  } else {
    result.average_factor = 1.0f;
  }
  result.bias = tl_model.param.global_bias;

  result.set_element_postproc(element_op::disable);
  result.row_postproc = row_op::disable;

  auto tl_pred_transform = std::string{tl_model.param.pred_transform};
  if (
      tl_pred_transform == std::string{"identity"} ||
      tl_pred_transform == std::string{"identity_multiclass"}) {
    result.set_element_postproc(element_op::disable);
    result.row_postproc = row_op::disable;
  } else if (tl_pred_transform == std::string{"signed_square"}) {
    result.set_element_postproc(element_op::signed_square);
  } else if (tl_pred_transform == std::string{"hinge"}) {
    result.set_element_postproc(element_op::hinge);
  } else if (tl_pred_transform == std::string{"sigmoid"}) {
    result.postproc_constant = tl_model.param.sigmoid_alpha;
    result.set_element_postproc(element_op::sigmoid);
  } else if (tl_pred_transform == std::string{"exponential"}) {
    result.set_element_postproc(element_op::exponential);
  } else if (tl_pred_transform == std::string{"exponential_standard_ratio"}) {
    result.postproc_constant = tl_model.param.ratio_c;
    result.set_element_postproc(element_op::exponential_standard_ratio);
  } else if (tl_pred_transform == std::string{"logarithm_one_plus_exp"}) {
    result.set_element_postproc(element_op::logarithm_one_plus_exp);
  } else if (tl_pred_transform == std::string{"max_index"}) {
    result.row_postproc = row_op::max_index;
  } else if (tl_pred_transform == std::string{"softmax"}) {
    result.row_postproc = row_op::softmax;
  } else if (tl_pred_transform == std::string{"multiclass_ova"}) {
    result.postproc_constant = tl_model.param.sigmoid_alpha;
    result.set_element_postproc(element_op::sigmoid);
  } else {
    throw unconvertible_model_exception{"Unrecognized Treelite pred_transform string"};
  }

  return result;
}

template <typename tl_threshold_t, typename tl_output_t, std::size_t variant_index>
auto convert_model(
    treelite::ModelImpl<tl_threshold_t, tl_output_t> const& tl_model,
    std::size_t target_variant_index) {
  auto result = tl_dispatched_model{};
  if constexpr (variant_index != std::variant_size_v<tl_dispatched_model>) {
    if (variant_index == target_variant_index) {
      using model_type = std::variant_alternative_t<variant_index, tl_dispatched_model>;
      if constexpr (
          std::is_same_v<tl_threshold_t, typename model_type::tree_type::node_type::value_type> &&
          (std::is_same_v<tl_output_t, typename model_type::tree_type::output_type> ||
          std::is_same_v<std::vector<tl_output_t>, typename model_type::tree_type::output_type>)) {
        result = convert_dispatched_model<variant_index, tl_threshold_t, tl_output_t>(tl_model);
      } else {
        throw unconvertible_model_exception("Unexpected TL types for this variant");
      }
    } else {
      result = convert_model<tl_threshold_t, tl_output_t, variant_index + 1>(tl_model, target_variant_index);
    }
  }
  return result;
}

template <typename tl_threshold_t, typename tl_output_t>
auto convert_model(treelite::ModelImpl<tl_threshold_t, tl_output_t> const& tl_model) {

  auto max_offset = std::accumulate(std::begin(tl_model.trees), std::end(tl_model.trees), int{}, [](auto&& prev_max, auto&& tree) {
    return std::max(prev_max, tree.num_nodes);
  });
  // TODO (wphicks): max_offset should be the min of the value calculated in
  // the above and 2**d + 1 where d is the max depth of any tree. For now, we
  // are just always using std::uint32_t for offset_t because using
  // std::uint16_t or lower will not reduce the overall size of the node with
  // padding.
  // https://github.com/triton-inference-server/fil_backend/issues/206

  auto constexpr large_threshold = std::size_t{std::is_same_v<tl_threshold_t, double>};
  auto const large_num_feature = std::size_t{tl_model.num_feature >= std::numeric_limits<std::uint16_t>::max()};
  auto const large_max_offset = std::size_t{max_offset >= std::numeric_limits<std::uint16_t>::max()};
  auto constexpr non_integer_output = std::size_t{!std::is_same_v<tl_output_t, std::uint32_t>};
  auto const has_vector_leaves = std::size_t{
    tl_model.task_param.leaf_vector_size > 1
  };

  auto variant_index = std::size_t{
    (large_threshold << 4) +
    (large_num_feature << 3) +
    (large_max_offset << 2) +
    (non_integer_output << 1) +
    has_vector_leaves
  };

  return convert_model<tl_threshold_t, tl_output_t, 0>(tl_model, variant_index);
}

}
