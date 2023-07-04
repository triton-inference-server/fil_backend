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

#include <cstddef>
#include <functional>
#include <herring/omp_helpers.hpp>
#include <herring/output_ops.hpp>
#include <herring/tree.hpp>
#include <herring/type_helpers.hpp>
#include <new>
#include <numeric>
#include <type_traits>
#include <vector>


namespace herring {

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
#else
auto constexpr hardware_constructive_interference_size = std::size_t{64};
#endif

template <
    typename value_t, typename feature_index_t, typename offset_t,
    typename output_index_t, typename output_t>
struct simple_model {
  using simple_tree_t =
      simple_tree<value_t, feature_index_t, offset_t, output_index_t, output_t>;
  using lookup_tree_t =
      lookup_tree<value_t, feature_index_t, offset_t, output_index_t, output_t>;
  using tree_type = std::conditional_t<
      std::is_same_v<value_t, output_t> ||
          std::is_same_v<output_index_t, output_t>,
      simple_tree_t, lookup_tree_t>;
  using sum_elem_type =
      typename is_container_specialization<output_t, std::vector>::value_type;

  std::vector<tree_type> trees;
  std::size_t num_class;
  std::size_t num_feature;
  row_op row_postproc;
  float average_factor;
  float bias;
  float postproc_constant;
  std::vector<bool> mutable row_has_missing;
  bool use_inclusive_threshold;
  bool has_categorical_trees;

  void predict(
      float const* input, std::size_t num_row, float* output,
      thread_count<int> nthread) const
  {
    // This dispatch structure is designed to determine as early as possible
    // whether a "slow" path is required and to convert booleans that
    // determine slow/fast path execution to compile-time constants such that
    // the compiled fast path never has to make checks required by the slow
    // path. Within the implementation of predict_, there is further
    // dispatching of the same sort to allow subsets of the model to use as
    // much fast-path coode as possible.
    //
    // TODO (wphicks): Much of this could be cleaned up with some template
    // metaprogramming and a few helper functions for switching to various
    // compile-time paths based on runtime boolean values.
    // (https://github.com/triton-inference-server/fil_backend/issues/205)
    //
    if (!precompute_missing(input, num_row)) {
      if (!use_inclusive_threshold) {
        if (!has_categorical_trees) {
          predict_<false, false, false>(input, output, num_row, nthread);
        } else {
          predict_<false, true, false>(input, output, num_row, nthread);
        }
      } else {
        if (!has_categorical_trees) {
          predict_<false, false, true>(input, output, num_row, nthread);
        } else {
          predict_<false, true, true>(input, output, num_row, nthread);
        }
      }
    } else {
      if (!use_inclusive_threshold) {
        if (!has_categorical_trees) {
          predict_<true, false, false>(input, output, num_row, nthread);
        } else {
          predict_<true, true, false>(input, output, num_row, nthread);
        }
      } else {
        if (!has_categorical_trees) {
          predict_<true, false, true>(input, output, num_row, nthread);
        } else {
          predict_<true, true, true>(input, output, num_row, nthread);
        }
      }
    }
  }

  void set_element_postproc(element_op element_postproc)
  {
    postprocess_element =
        [this,
         element_postproc]() -> std::function<void(sum_elem_type, float*)> {
      auto constant = postproc_constant;
      switch (element_postproc) {
        case element_op::signed_square:
          return [](sum_elem_type elem, float* out) {
            *out = std::copysign(elem * elem, elem);
          };
        case element_op::hinge:
          return [](sum_elem_type elem, float* out) {
            *out = elem > sum_elem_type{} ? sum_elem_type{1} : sum_elem_type{0};
          };
        case element_op::sigmoid:
          return [constant](sum_elem_type elem, float* out) {
            *out = sum_elem_type{1} /
                   (sum_elem_type{1} + std::exp(-constant * elem));
          };
        case element_op::exponential:
          return [](sum_elem_type elem, float* out) { *out = std::exp(elem); };
        case element_op::exponential_standard_ratio:
          return [constant](sum_elem_type elem, float* out) {
            *out = std::exp(-elem / constant);
          };
        case element_op::logarithm_one_plus_exp:
          return [](sum_elem_type elem, float* out) {
            *out = std::log1p(std::exp(elem));
          };
        default:
          return [](sum_elem_type elem, float* out) { *out = elem; };
      }
    }();
  }

 private:
  std::function<void(sum_elem_type, float*)> postprocess_element;

  auto precompute_missing(float const* input, std::size_t num_row) const
  {
    auto result = false;
    if (num_row > row_has_missing.size()) {
      row_has_missing.resize(num_row);
    }
    for (auto row_index = std::size_t{}; row_index < num_row; ++row_index) {
      row_has_missing[row_index] = std::transform_reduce(
          input + row_index * num_feature,
          input + (row_index + 1) * num_feature, false, std::logical_or<>(),
          [](auto val) { return std::isnan(val); });
      result = result || row_has_missing[row_index];
    }
    return result;
  }

  void apply_postprocessing(
      std::vector<sum_elem_type> const& grove_sum, float* output,
      std::size_t num_row, std::size_t num_grove,
      thread_count<int> nthread) const
  {
    if (row_postproc != row_op::max_index) {
#pragma omp parallel for num_threads(static_cast<int>(nthread))
      for (auto row_index = std::size_t{}; row_index < num_row; ++row_index) {
        auto const grove_output_index = row_index * num_class * num_grove;
        for (auto class_index = std::size_t{}; class_index < num_class;
             ++class_index) {
          auto const class_output_index =
              grove_output_index + class_index * num_grove;
          auto const grove_sum_begin = std::begin(grove_sum);
          postprocess_element(
              std::reduce(
                  grove_sum_begin + class_output_index,
                  grove_sum_begin + class_output_index + num_grove,
                  bias * average_factor) /
                  average_factor,
              output + row_index * num_class + class_index);
        }
        if (row_postproc == row_op::softmax) {
          auto const row_begin = output + row_index * num_class;
          auto const row_end = row_begin + num_class;
          auto const max_value = *std::max_element(row_begin, row_end);
          std::transform(
              row_begin, row_end, row_begin, [&max_value](auto const& val) {
                return std::exp(val - max_value);
              });
          auto const normalization = std::reduce(row_begin, row_end);
          std::transform(
              row_begin, row_end, output + row_index * num_class,
              [&normalization](auto const& val) {
                return val / normalization;
              });
        }
      }
    } else {
#pragma omp parallel for num_threads(static_cast<int>(nthread))
      for (auto row_index = std::size_t{}; row_index < num_row; ++row_index) {
        auto grove_output_index = row_index * num_class * num_grove;
        auto row_output = std::vector<float>(num_class, 0);

        for (auto class_index = std::size_t{}; class_index < num_class;
             ++class_index) {
          auto class_output_index =
              grove_output_index + class_index * num_grove;
          auto grove_sum_begin = std::begin(grove_sum);
          postprocess_element(
              std::reduce(
                  grove_sum_begin + class_output_index,
                  grove_sum_begin + class_output_index + num_grove,
                  bias * average_factor) /
                  average_factor,
              &(row_output[class_index]));
        }
        output[row_index] = std::distance(
            std::begin(row_output),
            std::max_element(std::begin(row_output), std::end(row_output)));
      }
    }
  }


  template <
      bool missing_value_in_input, bool categorical_model,
      bool inclusive_threshold>
  void predict_(
      float const* input, float* output, std::size_t num_row,
      thread_count<int> nthread) const
  {
    // "Groves" are groups of trees which are processed together in a single
    // thread. Similarly, "blocks" are groups of rows that are processed
    // together

    // Align grove boundaries on cache lines
    auto constexpr grove_size = hardware_constructive_interference_size;
    // Align block boundaries on cache lines
    auto constexpr block_size = hardware_constructive_interference_size;

    auto const num_tree = trees.size();
    auto const num_grove =
        (num_tree / grove_size + (num_tree % grove_size != 0));
    auto const num_block = (num_row / block_size + (num_row % block_size != 0));

    auto forest_sum = std::vector<sum_elem_type>(
        num_row * num_class * num_grove, sum_elem_type{});

#pragma omp parallel for num_threads(static_cast<int>(nthread))
    for (auto task_index = std::size_t{}; task_index < num_grove * num_block;
         ++task_index) {
      auto const grove_index = task_index / num_block;
      auto const block_index = task_index % num_block;

      auto const starting_row = block_index * block_size;
      auto const max_row = std::min(starting_row + block_size, num_row);
      for (auto row_index = starting_row; row_index < max_row; ++row_index) {
        auto const starting_tree = grove_index * grove_size;
        auto const max_tree = std::min(starting_tree + grove_size, num_tree);
        for (auto tree_index = starting_tree; tree_index < max_tree;
             ++tree_index) {
          auto const& tree = trees[tree_index];

          // Find leaf node
          auto node_index = std::size_t{};
          while (tree.nodes[node_index].distant_offset != 0) {
            if constexpr (missing_value_in_input) {
              if (not row_has_missing[row_index]) {
                node_index += tree.template evaluate_tree_node<
                    false, categorical_model, inclusive_threshold>(
                    node_index, input + row_index * num_feature);
              } else {
                node_index += tree.template evaluate_tree_node<
                    true, categorical_model, inclusive_threshold>(
                    node_index, input + row_index * num_feature);
              }
            } else {
              node_index += tree.template evaluate_tree_node<
                  false, categorical_model, inclusive_threshold>(
                  node_index, input + row_index * num_feature);
            }
          }

          // Add leaf contribution to output
          if constexpr (is_container_specialization<
                            output_t, std::vector>::value) {
            auto leaf_output = tree.get_leaf_value(node_index);
            for (auto class_index = std::size_t{}; class_index < num_class;
                 ++class_index) {
              forest_sum
                  [row_index * num_class * num_grove + class_index * num_grove +
                   grove_index] += leaf_output[class_index];
            }
          } else {
            auto class_index = tree_index % num_class;
            auto cur_index = row_index * num_class * num_grove +
                             class_index * num_grove + grove_index;
            forest_sum
                [row_index * num_class * num_grove + class_index * num_grove +
                 grove_index] += tree.get_leaf_value(node_index);
          }
        }  // Trees in grove
      }    // Rows in block
    }      // Tasks (groves x blocks)

    apply_postprocessing(forest_sum, output, num_row, num_grove, nthread);
  }
};
}  // namespace herring
