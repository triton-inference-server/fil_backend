#pragma once
#include <cstddef>
#include <queue>
#include <stack>
#include <treelite/tree.h>
#include <treelite/typeinfo.h>
#include <herring2/decision_forest.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/tree_layout.hpp>

namespace herring {

using kayak::raw_index_t;

struct model_import_error : std::exception {
  model_import_error() : model_import_error("Error while importing model") {}
  model_import_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template <kayak::tree_layout layout, typename T>
struct traversal_container {
  using backing_container_t = std::conditional_t<
    layout == kayak::tree_layout::depth_first,
    std::stack<T>,
    std::queue<T>
  >;
  void add(T const& val) {
    data_.push(val);
  }
  auto next() {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      auto result = data_.top();
      data_.pop();
      return result;
    } else {
      auto result = data_.front();
      data_.pop();
      return result;
    }
  }
  auto peek() {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      return data_.top();
    } else {
      return data_.front();
    }
  }
  [[nodiscard]] auto empty() {
    return data_.empty();
  }
  auto size() {
    return data_.size();
  }
 private:
  backing_container_t data_;
};

template<kayak::tree_layout layout>
struct treelite_importer {
  template<typename tl_threshold_t, typename tl_output_t>
  struct treelite_node {
    treelite::Tree<tl_threshold_t, tl_output_t> const& tree;
    int node_id;
    std::size_t parent_index;
    std::size_t own_index;

    auto is_categorical() {
      return tree.SplitType(node_id) == treelite::SplitFeatureType::kCategorical;
    }

    auto categories() {
      auto result = decltype(tree.MatchingCategories(node_id)){};
      if (is_categorical()) {
        result = tree.MatchingCategories(node_id);
      }
      return result;
    }

    auto is_inclusive() {
      auto tl_operator = tree.ComparisonOp(node_id);
      return tl_operator == treelite::Operator::kGE || tl_operator == treelite::Operator::kLE;
    }
  };

  template<typename tl_threshold_t, typename tl_output_t, typename lambda_t>
  void node_for_each(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, lambda_t&& lambda) {
    using node_index_t = decltype(tl_tree.LeftChild());
    auto to_be_visited = traversal_container<layout, node_index_t>{};
    to_be_visited.add(node_index_t{});

    auto parent_indices = traversal_container<layout, std::size_t>{};
    auto cur_index = std::size_t{};
    parent_indices.add(cur_index);

    while (!to_be_visited.empty()) {
      auto node_id = to_be_visited.next();
      auto remaining_size = to_be_visited.size();

      auto tl_node = treelite_node{tl_tree, node_id, parent_indices.next(), cur_index};
      lambda(tl_node);

      if (!tl_tree.IsLeaf(node_id)) {
        auto tl_left_id = tl_tree.LeftChild(node_id);
        auto tl_right_id = tl_tree.RightChild(node_id);
        auto tl_operator = tl_tree.ComparisonOp(node_id);
        if (!tl_node.is_categorical()) {
          if (tl_operator == treelite::Operator::kLT || tl_operator == treelite::Operator::kLE) {
            to_be_visited.add(tl_left_id);
            to_be_visited.add(tl_right_id);
          } else if (tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kGE) {
            to_be_visited.add(tl_right_id);
            to_be_visited.add(tl_left_id);
          } else {
            throw model_import_error("Unrecognized Treelite operator");
          }
        } else {
          if (tl_tree.CategoriesListRightChild(node_id)) {
            to_be_visited.add(tl_right_id);
            to_be_visited.add(tl_left_id);
          } else {
            to_be_visited.add(tl_left_id);
            to_be_visited.add(tl_right_id);
          }
        }
        parent_indices.add(cur_index);
        parent_indices.add(cur_index);
      }
      ++cur_index;
    }
  }

  template<typename tl_threshold_t, typename tl_output_t, typename iter_t, typename lambda_t>
  void node_transform(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, iter_t& output_iter, lambda_t&& lambda) {
    node_for_each(
      tl_tree,
      [&output_iter, &lambda](auto&& tl_node) {
        *output_iter = lambda(tl_node);
        ++output_iter;
      }
    );
  }

  template<typename tl_threshold_t, typename tl_output_t, typename T, typename lambda_t>
  void node_accumulate(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, T init, lambda_t&& lambda) {
    auto result = init;
    node_for_each(
      tl_tree,
      [&result, &lambda](auto&& tl_node) {
        result = lambda(result, tl_node);
      }
    );
    return result;
  }

  template<typename tl_threshold_t, typename tl_output_t>
  auto get_nodes(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree) {
    auto result = std::vector<treelite_node<tl_threshold_t, tl_output_t>>{};
    result.reserve(tl_tree.num_nodes);
    node_transform(tl_tree, std::back_inserter(result), [](auto&& node) { return node; });
    return result;
  }

  template<typename tl_threshold_t, typename tl_output_t>
  auto get_offsets(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree) {
    auto result = std::vector<raw_index_t>{tl_tree.num_nodes};
    auto nodes = get_nodes(tl_tree);
    try {
      for (auto i = std::size_t{}; i < nodes.size(); ++i) {
        // Current index should always be greater than or equal to parent index.
        // Later children will overwrite values set by earlier children, ensuring
        // that most distant offset is used.
        result[nodes[i].parent_index] = kayak::detail::index_type<true>{i - nodes[i].parent_index};
      }
    } catch(kayak::detail::bad_index const& err) {
      throw model_import_error(
        "Offset between parent and child node was invalid or exceeded maximum allowed value"
      );
    }

    return result;
  }

  template<typename lambda_t>
  void tree_for_each(treelite::Model const& tl_model, lambda_t&& lambda) {
    tl_model.Dispatch([&lambda](auto&& concrete_tl_model) {
      std::for_each(
        std::begin(concrete_tl_model.trees),
        std::end(concrete_tl_model.trees),
        lambda
      );
    });
  }

  template<typename iter_t, typename lambda_t>
  void tree_transform(treelite::Model const& tl_model, iter_t output_iter, lambda_t&& lambda) {
    tl_model.Dispatch([&output_iter, &lambda](auto&& concrete_tl_model) {
      std::transform(
        std::begin(concrete_tl_model.trees),
        std::end(concrete_tl_model.trees),
        output_iter,
        lambda
      );
    });
  }

  template<typename T, typename lambda_t>
  void tree_accumulate(treelite::Model const& tl_model, T init, lambda_t&& lambda) {
    auto result = init;
    tree_for_each(
      tl_model,
      [&result, &lambda](auto&& tree) {
        result = lambda(result, tree);
      }
    );
    return result;
  }

  auto num_trees(treelite::Model const& tl_model) {
    auto result = std::size_t{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.trees.size();
    });
    return result;
  }

  auto get_offsets(treelite::Model const& tl_model) {
    auto result = std::vector<std::vector<raw_index_t>>{};
    result.reserve(num_trees(tl_model));
    tree_transform(tl_model, std::back_inserter(result), [this](auto&&tree) {
      return get_offsets(tree);
    });
  }

  auto get_tree_sizes(treelite::Model const& tl_model) {
    auto result = std::vector<raw_index_t>{};
    tree_transform(
      tl_model,
      std::back_inserter(result),
      [](auto&& tree) { return tree.num_nodes; }
    );
    return result;
  }

  auto get_num_class(treelite::Model const& tl_model) {
    auto result = kayak::detail::index_type<true>{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.task_param.num_class;
    });
    return result;
  }

  auto get_num_feature(treelite::Model const& tl_model) {
    auto result = kayak::detail::index_type<true>{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.num_feature;
    });
    return result;
  }

  auto get_max_num_categories(treelite::Model const& tl_model) {
    return tree_accumulate(tl_model, raw_index_t{}, [this](auto&& accum, auto&& tree) {
      node_accumulate(tree, accum, [](auto&& cur_accum, auto&& tl_node) {
        auto result = cur_accum;
        for (auto&& cat : tl_node.categories()) {
          result = (cat + 1 > result) ? cat + 1 : result;
        }
        return result;
      });
    });

  }

  auto uses_double_thresholds(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = true;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_double_outputs(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = true;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      case treelite::TypeInfo::kUInt32:
        result = false;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_integer_outputs(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = false;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      case treelite::TypeInfo::kUInt32:
        result = true;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto import(treelite::Model const& tl_model, kayak::detail::index_type<true> align_bytes = raw_index_t{}) {
    auto result = forest_model_variant{};
    auto num_feature = get_num_feature(tl_model);
    auto max_num_categories = get_max_num_categories(tl_model);
    auto use_double_thresholds = uses_double_thresholds(tl_model);
    auto use_double_outputs = uses_double_outputs(tl_model);
    auto use_integer_outputs = uses_integer_outputs(tl_model);

    auto offsets = get_offsets(tl_model);
    auto tree_sizes = std::vector<raw_index_t>{};
    try {
      std::transform(
        std::begin(offsets),
        std::end(offsets),
        std::back_inserter(tree_sizes),
        [](auto&& tree_offsets) {
          return kayak::detail::index_type<true>{tree_offsets.size()};
        }
      );
    } catch(kayak::detail::bad_index const& err) {
      throw model_import_error(
        "Tree too large to be represented in Herring format"
      );
    }
    return result;
  }

};

}
