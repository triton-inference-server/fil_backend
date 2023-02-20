/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <treeshap_model.h>
#include <linear_treeshap_constants.h>
#include <names.h>
#include <tl_model.h>

#include <cstddef>
#include <float.h>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>
#include "herring/tl_helpers.hpp"

namespace triton { namespace backend { namespace NAMESPACE {

// The linear treeshap algorithm requires some extra info for each tree
template<typename tl_threshold_t, typename tl_output_t>
struct TreeMetaInfo{
  std::vector<float> weights;
  std::vector<int> edge_heights;
  std::vector<int> parents;
  int max_depth = 0;
  const treelite::Tree<tl_threshold_t, tl_output_t>& tree;

  int Recurse(int node, int parent = -1, int depth = 0, std::map<int, int> seen_features = std::map<int, int>())
  {
    if (node != 0) {
      auto feature = tree.SplitIndex(parent);
      float weight = 0.0f;
      if (tree.HasSumHess(parent) && tree.HasSumHess(node)) {
        weight =
            static_cast<double>(tree.SumHess(node) / tree.SumHess(parent));
      } else if (tree.HasDataCount(parent) && tree.HasDataCount(node)) {
        weight =
            static_cast<double>(tree.DataCount(node)) / tree.DataCount(parent);
      } else {
        throw rapids::TritonException(
            rapids::Error::Unsupported, "Model does not have node statistics.");
      }
      if (seen_features.count(feature)) {
        auto seen_node_id = seen_features[feature];
        parents[node] = seen_node_id;
        weight *= weights[seen_node_id];
      }
      weights[node] = weight;
      seen_features[feature] = node;
    }

    if (!tree.IsLeaf(node)) {
      auto left_max_features = Recurse(tree.LeftChild(node),node, depth + 1, seen_features);
      auto right_max_features = Recurse(tree.RightChild(node), node,depth + 1, seen_features);
      edge_heights[node] = std::max(left_max_features, right_max_features);

    } else {
      edge_heights[node] = seen_features.size();
      max_depth = std::max(max_depth, depth);
    }
    return edge_heights[node];
  }

  explicit TreeMetaInfo(const treelite::Tree<tl_threshold_t, tl_output_t>& tree)
      : tree(tree)
  {
    weights.resize(tree.num_nodes, 1.0f);
    parents.resize(tree.num_nodes, -1);
    edge_heights.resize(tree.num_nodes);
    Recurse(0);
  }
};

// Helper to strip types from treelite
template<typename tl_threshold_t, typename tl_output_t>
auto get_tree_meta_info_vector(const treelite::ModelImpl<tl_threshold_t, tl_output_t>&){
  return std::vector<TreeMetaInfo<tl_threshold_t, tl_output_t>>{};
}

float psi(float *e, const float *offset, const float *Base, float q, const float *n, int d)
{
    float res = 0.;
    for (int i = 0; i < d; i++)
    {
        res += e[i] *offset[i] / (Base[i] + q) * n[i];
    }
    return res / d;
}

void times(const float *input, float *output, float scalar, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = input[i] * scalar;
    }
};

void times_broadcast(const float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] *= input[i];
    }
};

void addition(float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] += input[i];
    }
};

void write(float *from, float *to, int size)
{
    for (int i = 0; i < size; i++)
    {
        to[i] = from[i];
    }
};



template <typename ThresholdType>
bool decision_non_categorical(float fvalue, ThresholdType threshold, treelite::Operator op) {
  // XGBoost
  if (op == treelite::Operator::kLT) {
    return fvalue < threshold;
  }
  // LightGBM, sklearn, cuML RF
  if (op == treelite::Operator::kLE) {
    return fvalue <= threshold;
  }
  switch (op) {
    case treelite::Operator::kEQ:
      return fvalue == threshold;
    case treelite::Operator::kGT:
      return fvalue > threshold;
    case treelite::Operator::kGE:
      return fvalue >= threshold;
    default:
      TREELITE_CHECK(false) << "Unrecognized comparison operator " << static_cast<int>(op);
      return -1;
  }
}

inline bool decision_categorical(float fvalue, const std::vector<std::uint32_t>& matching_categories,
                               bool categories_list_right_child) {
  bool is_matching_category;
  auto max_representable_int = static_cast<float>(std::uint32_t(1) << FLT_MANT_DIG);
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    is_matching_category = false;
  } else {
    const auto category_value = static_cast<std::uint32_t>(fvalue);
    is_matching_category = (
        std::find(matching_categories.begin(), matching_categories.end(), category_value)
        != matching_categories.end());
  }
  if (categories_list_right_child) {
    return !is_matching_category;
  } else {
    return is_matching_category;
  }
}

template <typename tl_threshold_t, typename tl_output_t>
bool
decision(
    const treelite::Tree<tl_threshold_t, tl_output_t>& tree, const float fvalue,
    int node)
{
  // Missing
  if(std::isnan(fvalue)){
    return tree.DefaultChild(node) == tree.LeftChild(node);
  }

  if (tree.SplitType(node) == treelite::SplitFeatureType::kCategorical) {
    return decision_categorical(
        fvalue, tree.MatchingCategories(node),
        tree.CategoriesListRightChild(node));
  } else {
    return decision_non_categorical(fvalue, tree.Threshold(node), tree.ComparisonOp(node));
  }
}

template <typename tl_threshold_t, typename tl_output_t>
float
leaf_probability(
    const treelite::Tree<tl_threshold_t, tl_output_t>& tree, int node)
{
  if (tree.HasSumHess(node) && tree.HasSumHess(0)) {
    return static_cast<double>(tree.SumHess(node) / tree.SumHess(0));
  } else if (tree.HasDataCount(node) && tree.HasDataCount(0)) {
    return static_cast<double>(tree.DataCount(node)) / tree.DataCount(0);
  }
}

template <typename tl_threshold_t, typename tl_output_t>
void inference(const TreeMetaInfo<tl_threshold_t, tl_output_t> &tree_info,
               const float *x,
               uint8_t *activation,
               float *value,
               float *C,
               float *E,
               int node = 0,
               int feature = -1,
               int depth = 0)
{
    const auto& tree = tree_info.tree;
    float s = 0.;
    int parent = tree_info.parents[node];
    if (parent >= 0)
    {
        activation[node] = activation[node] & activation[parent];
        if (activation[parent]) {
          s = 1 / tree_info.weights[parent];
        }
    }

    float *current_e = E + depth * tree_info.max_depth;
    float *child_e = E + (depth + 1) * tree_info.max_depth;
    float *current_c = C + depth * tree_info.max_depth;
    float q = 0.;
    if (feature >= 0) {
      if (activation[node]) {
        q = 1 / tree_info.weights[node];
      }

      float *prev_c = C + (depth - 1) * tree_info.max_depth;
      for (int i = 0; i < tree_info.max_depth; i++) {
        current_c[i] = prev_c[i] * (kBase[i] + q);
      }

      if (parent >= 0) {
        for (int i = 0; i < tree_info.max_depth; i++) {
          current_c[i] = current_c[i] / (kBase[i] + s);
        }
      }
    }
    int offset_degree = 0;
    int left = tree.LeftChild(node);
    int right = tree.RightChild(node);
    if (!tree.IsLeaf(node)) {
      activation[left] = decision(tree, x[tree.SplitIndex(node)], node);
      activation[right] = !activation[left];
      inference(
          tree_info, x, activation, value, C, E, left, tree.SplitIndex(node),
          depth + 1);
      offset_degree =
          tree_info.edge_heights[node] - tree_info.edge_heights[left];
      times_broadcast(kOffset[offset_degree], child_e, tree_info.max_depth);
      write(child_e, current_e, tree_info.max_depth);
      inference(
          tree_info, x, activation, value, C, E, right, tree.SplitIndex(node),
          depth + 1);
      offset_degree =
          tree_info.edge_heights[node] - tree_info.edge_heights[right];
      times_broadcast(kOffset[offset_degree], child_e, tree_info.max_depth);
      addition(child_e, current_e, tree_info.max_depth);
    } else {

      times(
          current_c, current_e, tree.LeafValue(node) * leaf_probability(tree, node),
          tree_info.max_depth);
    }

    if (feature >= 0) {
      if (parent >= 0 && !activation[parent]) {
        return;
      }

      value[feature] +=
          (q - 1) * psi(current_e, kOffset[0], kBase, q, kNorm[tree_info.edge_heights[node]],
                        tree_info.edge_heights[node]);
      if (parent >= 0) {
        offset_degree = tree_info.edge_heights[parent] - tree_info.edge_heights[node];
        value[feature] -=
            (s - 1) * psi(current_e, kOffset[offset_degree], kBase, s,
                          kNorm[tree_info.edge_heights[parent]], tree_info.edge_heights[parent]);
      }
    }
};

// Shapley values for a single instance/tree
template <typename tl_threshold_t, typename tl_output_t>
void
linear_treeshap(
    const TreeMetaInfo<tl_threshold_t, tl_output_t>& tree_info,
    float* output, float const* input)
{
  int size = (tree_info.max_depth + 1) * tree_info.max_depth;
  std::vector<float> C(size, 1);
  std::vector<float> E(size);
  std::vector<uint8_t> activation(tree_info.tree.num_nodes);
  inference(tree_info, input, activation.data(), output, C.data(), E.data());
}

template <>
struct TreeShapModel<rapids::HostMemory> {
  TreeShapModel() = default;
  TreeShapModel(
      std::shared_ptr<TreeliteModel> tl_model):tl_model_(tl_model)
  {

    
  }


  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t n_rows, std::size_t n_cols) const
  {
      tl_model_->base_tl_model()->Dispatch([&](const auto& model) {
        auto info = get_tree_meta_info_vector(model);
        info.reserve(model.trees.size());
        for (auto tree_idx = 0; tree_idx < model.trees.size(); ++tree_idx) {
          info.push_back(
              typename decltype(info)::value_type(model.trees[tree_idx]));
        }
        for (auto tree_idx = 0; tree_idx < model.trees.size(); tree_idx++) {
          for (auto i = 0; i < n_rows; i++) {
            linear_treeshap(
                info[tree_idx], output.data() + i * (n_cols + 1),
                input.data() + i * n_cols);
          }
        }
        // Scale output
        auto scale = 1.0f / herring::get_average_factor(model);
        for (auto i = 0; i < output.size(); i++) {
          output.data()[i] = output.data()[i] * scale + model.param.global_bias;
        }
      });
  }
  std::shared_ptr<TreeliteModel> tl_model_;
};
}}}  // namespace triton::backend::NAMESPACE
