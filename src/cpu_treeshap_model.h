/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <float.h>
#include <linear_treeshap_constants.h>
#include <names.h>
#include <tl_model.h>
#include <treeshap_model.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>

#include "herring/tl_helpers.hpp"

namespace triton { namespace backend { namespace NAMESPACE {

template <typename tl_threshold_t, typename tl_output_t>
double
leaf_probability(
    const treelite::Tree<tl_threshold_t, tl_output_t>& tree, int node)
{
  if (tree.HasSumHess(node) && tree.HasSumHess(0)) {
    return static_cast<double>(tree.SumHess(node) / tree.SumHess(0));
  } else if (tree.HasDataCount(node) && tree.HasDataCount(0)) {
    return static_cast<double>(tree.DataCount(node)) / tree.DataCount(0);
  }
  return 0.0f;
}

// The linear treeshap algorithm requires some extra info for each tree
template <typename tl_threshold_t, typename tl_output_t>
struct TreeMetaInfo {
  std::vector<double> weights;
  std::vector<int> edge_heights;
  std::vector<int> parents;
  int max_depth = 0;
  const treelite::Tree<tl_threshold_t, tl_output_t>& tree;
  double global_bias;
  int class_idx;

  std::pair<int, double> Recurse(
      int node, int parent = -1, int depth = 0,
      std::map<int, int> seen_features = std::map<int, int>())
  {
    if (node != 0) {
      auto feature = tree.SplitIndex(parent);
      double weight = 0.0;
      if (tree.HasSumHess(parent) && tree.HasSumHess(node)) {
        weight = static_cast<double>(tree.SumHess(node) / tree.SumHess(parent));
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

    double bias = 0.0;
    if (!tree.IsLeaf(node)) {
      auto [left_max_features, left_bias] =
          Recurse(tree.LeftChild(node), node, depth + 1, seen_features);
      auto [right_max_features, right_bias] =
          Recurse(tree.RightChild(node), node, depth + 1, seen_features);
      edge_heights[node] = std::max(left_max_features, right_max_features);
      bias = left_bias + right_bias;
    } else {
      edge_heights[node] = seen_features.size();
      max_depth = std::max(max_depth, depth);
      double leaf_value = tree.HasLeafVector(node)
                              ? tree.LeafVector(node)[class_idx]
                              : tree.LeafValue(node);
      bias = leaf_probability(tree, node) * leaf_value;
    }
    return std::make_pair(edge_heights[node], bias);
  }

  explicit TreeMetaInfo(
      const treelite::Tree<tl_threshold_t, tl_output_t>& tree, int class_idx)
      : tree(tree), class_idx(class_idx)
  {
    weights.resize(tree.num_nodes, 1.0f);
    parents.resize(tree.num_nodes, -1);
    edge_heights.resize(tree.num_nodes);
    auto [_, bias] = Recurse(0);
    global_bias = bias;
    if (max_depth >= 32) {
      throw rapids::TritonException(
          rapids::Error::Unsupported,
          "Tree depths above 32 not supported. Shap cannot be considered "
          "accurate at this depth.");
    }
  }
};

// Helper to get types from treelite
template <typename tl_threshold_t, typename tl_output_t>
auto
get_tree_meta_info_vector(
    const treelite::ModelImpl<tl_threshold_t, tl_output_t>&)
{
  return std::vector<TreeMetaInfo<tl_threshold_t, tl_output_t>>{};
}

auto
psi(double* e, const double* offset, const double* Base, double q,
    const double* n, int d)
{
  double res = 0.;
  for (int i = 0; i < d; i++) {
    res += e[i] * offset[i] / (Base[i] + q) * n[i];
  }
  return res / d;
}

template <typename ThresholdType>
bool
decision_non_categorical(
    float fvalue, ThresholdType threshold, treelite::Operator op)
{
  switch (op) {
    case treelite::Operator::kLT:
      return fvalue < threshold;
    case treelite::Operator::kLE:
      return fvalue <= threshold;
    case treelite::Operator::kEQ:
      return fvalue == threshold;
    case treelite::Operator::kGT:
      return fvalue > threshold;
    case treelite::Operator::kGE:
      return fvalue >= threshold;
    default:
      TREELITE_CHECK(false)
          << "Unrecognized comparison operator " << static_cast<int>(op);
      return -1;
  }
}

inline bool
decision_categorical(
    float fvalue, const std::vector<std::uint32_t>& matching_categories,
    bool categories_list_right_child)
{
  bool is_matching_category;
  auto max_representable_int =
      static_cast<float>(std::uint32_t(1) << FLT_MANT_DIG);
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    is_matching_category = false;
  } else {
    const auto category_value = static_cast<std::uint32_t>(fvalue);
    is_matching_category =
        (std::find(
             matching_categories.begin(), matching_categories.end(),
             category_value) != matching_categories.end());
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
  if (std::isnan(fvalue)) {
    return tree.DefaultChild(node) == tree.LeftChild(node);
  }

  if (tree.SplitType(node) == treelite::SplitFeatureType::kCategorical) {
    return decision_categorical(
        fvalue, tree.MatchingCategories(node),
        tree.CategoriesListRightChild(node));
  } else {
    return decision_non_categorical(
        fvalue, tree.Threshold(node), tree.ComparisonOp(node));
  }
}


template <typename tl_threshold_t, typename tl_output_t>
void
inference(
    const TreeMetaInfo<tl_threshold_t, tl_output_t>& tree_info, const float* x,
    uint8_t* activation, float* value, double* C, double* E, int node = 0,
    int feature = -1, int depth = 0)
{
  const auto& tree = tree_info.tree;
  double s = 0.;
  int parent = tree_info.parents[node];
  if (parent >= 0) {
    activation[node] = activation[node] & activation[parent];
    if (activation[parent]) {
      s = 1 / tree_info.weights[parent];
    }
  }

  double* current_e = E + depth * tree_info.max_depth;
  double* child_e = E + (depth + 1) * tree_info.max_depth;
  double* current_c = C + depth * tree_info.max_depth;
  double q = 0.;
  if (feature >= 0) {
    if (activation[node]) {
      q = 1 / tree_info.weights[node];
    }

    double* prev_c = C + (depth - 1) * tree_info.max_depth;
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
    offset_degree = tree_info.edge_heights[node] - tree_info.edge_heights[left];
    std::transform(
        kOffset[offset_degree], kOffset[offset_degree] + tree_info.max_depth,
        child_e, child_e, std::multiplies<double>());
    std::copy(child_e, child_e + tree_info.max_depth, current_e);
    inference(
        tree_info, x, activation, value, C, E, right, tree.SplitIndex(node),
        depth + 1);
    offset_degree =
        tree_info.edge_heights[node] - tree_info.edge_heights[right];
    std::transform(
        kOffset[offset_degree], kOffset[offset_degree] + tree_info.max_depth,
        child_e, child_e, std::multiplies<double>());
    std::transform(
        child_e, child_e + tree_info.max_depth, current_e, current_e,
        std::plus<double>());
  } else {
    double leaf_value = tree.HasLeafVector(node)
                            ? tree.LeafVector(node)[tree_info.class_idx]
                            : tree.LeafValue(node);
    std::transform(
        current_c, current_c + tree_info.max_depth, current_e,
        [&](auto&& elem) {
          return leaf_value * leaf_probability(tree, node) * elem;
        });
  }

  if (feature >= 0) {
    if (parent >= 0 && !activation[parent]) {
      return;
    }

    value[feature] += (q - 1) * psi(current_e, kOffset[0], kBase, q,
                                    kNorm[tree_info.edge_heights[node]],
                                    tree_info.edge_heights[node]);
    if (parent >= 0) {
      offset_degree =
          tree_info.edge_heights[parent] - tree_info.edge_heights[node];
      value[feature] -= (s - 1) * psi(current_e, kOffset[offset_degree], kBase,
                                      s, kNorm[tree_info.edge_heights[parent]],
                                      tree_info.edge_heights[parent]);
    }
  }
};

// Shapley values for a single instance/tree
template <typename tl_threshold_t, typename tl_output_t>
void
linear_treeshap(
    const TreeMetaInfo<tl_threshold_t, tl_output_t>& tree_info, float* output,
    float const* input, std::size_t n_cols)
{
  output[n_cols] += tree_info.global_bias;
  int size = (tree_info.max_depth + 1) * tree_info.max_depth;
  std::vector<double> C(size, 1);
  std::vector<double> E(size);
  std::vector<uint8_t> activation(tree_info.tree.num_nodes);
  inference(tree_info, input, activation.data(), output, C.data(), E.data());
}

template <>
struct TreeShapModel<rapids::HostMemory> {
  TreeShapModel() = default;
  TreeShapModel(std::shared_ptr<TreeliteModel> tl_model) : tl_model_(tl_model)
  {
    tl_model_->base_tl_model()->Dispatch([&](const auto& model) {
      meta_infos_ = get_tree_meta_info_vector(model);
      auto& info =
          std::get<decltype(get_tree_meta_info_vector(model))>(meta_infos_);

      num_class_ = std::max(
          model.task_param.num_class, model.task_param.leaf_vector_size);
      bool is_vector_leaf = model.task_param.leaf_vector_size > 1;
      int num_info =
          is_vector_leaf ? model.trees.size() * num_class_ : model.trees.size();
      info.reserve(num_info);
      // Deal with vector leaf models by duplicating the meta info structs once
      // for each class of a tree
      for (auto info_idx = 0; info_idx < num_info; info_idx++) {
        auto class_idx = info_idx % num_class_;
        auto tree_idx = is_vector_leaf ? info_idx / num_class_ : info_idx;
        info.push_back(
            typename std::remove_reference<decltype(info)>::type::value_type(
                model.trees.at(tree_idx), class_idx));
      }
      average_factor_ = herring::get_average_factor(model);
      global_bias_ = model.param.global_bias;
    });
  }

  void predict(
      rapids::Buffer<float>& output, rapids::Buffer<float const> const& input,
      std::size_t n_rows, std::size_t n_cols) const
  {
    thread_count<int> nthread(tl_model_->config().cpu_nthread);
    std::visit(
        [&](const auto& info) {
#pragma omp parallel for num_threads(static_cast<int>(nthread))
          for (auto i = 0; i < n_rows; i++) {
            for (auto info_idx = 0; info_idx < info.size(); info_idx++) {
              // One class per tree
              auto output_offset =
                  output.data() +
                  (i * num_class_ + info[info_idx].class_idx) * (n_cols + 1);
              linear_treeshap(
                  info[info_idx], output_offset, input.data() + i * n_cols,
                  n_cols);
            }
          }
          // Scale output
          auto scale = 1.0f / average_factor_;
          for (auto i = 0; i < output.size(); i++) {
            output.data()[i] = output.data()[i] * scale;
          }
          // Add global bias to bias column
          for (auto i = 0; i < n_rows; i++) {
            for (auto class_idx = 0; class_idx < num_class_; class_idx++) {
              auto output_offset =
                  (i * num_class_ + class_idx) * (n_cols + 1) + n_cols;
              output.data()[output_offset] += global_bias_;
            }
          }
        },
        meta_infos_);
  }
  std::shared_ptr<TreeliteModel> tl_model_;
  int num_class_;
  float average_factor_;
  float global_bias_;
  // Vector of supplementary information for each tree
  // Contains precomputed data necessary for linear treeshap algorithm
  std::variant<
      std::vector<TreeMetaInfo<float, float>>,
      std::vector<TreeMetaInfo<float, uint32_t>>,
      std::vector<TreeMetaInfo<double, uint32_t>>,
      std::vector<TreeMetaInfo<double, double>>>
      meta_infos_;
};
}}}  // namespace triton::backend::NAMESPACE
