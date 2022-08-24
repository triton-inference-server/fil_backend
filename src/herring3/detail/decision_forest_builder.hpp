#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdint.h>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>
#include <herring3/postproc_ops.hpp>
#include <herring3/detail/forest.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>
#include <kayak/detail/index_type.hpp>

namespace herring {
namespace detail {

struct model_builder_error : std::exception {
  model_builder_error() : model_builder_error("Error while building model") {}
  model_builder_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template<typename decision_forest_t>
struct decision_forest_builder {

  using node_type = typename decision_forest_t::node_type;

  void start_new_tree() {
    if (root_node_indexes_.size() == std::size_t{}) {
      root_node_indexes_.emplace_back();
    } else {
      max_tree_size_ = std::max(cur_tree_size_, max_tree_size_);
      if (alignment_ != std::size_t{}) {
        if (cur_tree_size_ % alignment_ != std::size_t{}) {
          auto padding = (alignment_ - cur_tree_size_ % alignment_);
          for (auto i = std::size_t{}; i < padding; ++i) {
            add_node(typename node_type::threshold_type{});
          }
        }
      }
      root_node_indexes_.push_back(
        root_node_indexes_.back() + cur_tree_size_
      );
      cur_tree_size_ = std::size_t{};
    }
  }

  template<typename value_t>
  void add_node(
    value_t val,
    bool is_leaf_node=true,
    bool default_to_distant_child=false,
    bool is_categorical_node=false,
    typename node_type::metadata_storage_type feature = typename node_type::metadata_storage_type{},
    typename node_type::offset_type offset = typename node_type::offset_type{1},
    bool is_inclusive=false
  ) {
    nodes_.emplace_back(
      val, is_leaf_node, default_to_distant_child, is_categorical_node, feature, offset
    );
    ++cur_tree_size_;
  }

  void set_element_postproc(element_op val) { element_postproc_ = val; }
  void set_row_postproc(row_op val) { row_postproc_ = val; }
  void set_average_factor(double val) { average_factor_ = val; }
  void set_bias(double val) { bias_ = val; }
  void set_postproc_constant(double val) { postproc_constant_ = val; }

  decision_forest_builder(std::size_t align_bytes=std::size_t{}) :
    cur_tree_size_{},
    alignment_{std::lcm(align_bytes, sizeof(node_type))},
    output_size_{1},
    element_postproc_{},
    average_factor_{},
    row_postproc_{},
    bias_{},
    postproc_constant_{},
    max_tree_size_{},
    nodes_{},
    root_node_indexes_{},
    vector_output_{} {
  }

  auto get_decision_forest(
      std::size_t num_feature,
      std::size_t num_class,
      kayak::device_type mem_type=kayak::device_type::cpu,
      int device=0,
      kayak::cuda_stream stream=kayak::cuda_stream{}
  ) {

    std::cout << "Max tree size: " << max_tree_size_ << "\n";
    // Allow narrowing for preprocessing constants. They are stored as doubles
    // for consistency in the builder but must be converted to the proper types
    // for the concrete forest model.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return decision_forest_t{
      kayak::buffer{
        kayak::buffer{nodes_.data(), nodes_.size()},
        mem_type,
        device,
        stream
      },
      kayak::buffer{
        kayak::buffer{root_node_indexes_.data(), root_node_indexes_.size()},
        mem_type,
        device,
        stream
      },
      num_feature,
      num_class,
      vector_output_.size() == 0 ?
        std::nullopt :
        std::make_optional<kayak::buffer<typename node_type::threshold_type>>(
          kayak::buffer{vector_output_.data(), vector_output_.size()},
          mem_type,
          device,
          stream
        ),
      output_size_,
      row_postproc_,
      element_postproc_,
      average_factor_,
      bias_,
      postproc_constant_
    };
#pragma GCC diagnostic pop
  }


 private:
  std::size_t cur_tree_size_;
  std::size_t alignment_;
  std::size_t output_size_;
  row_op row_postproc_;
  element_op element_postproc_;
  double average_factor_;
  double bias_;
  double postproc_constant_;
  std::size_t max_tree_size_;

  std::vector<node_type> nodes_;
  std::vector<std::size_t> root_node_indexes_;
  std::vector<typename node_type::threshold_type> vector_output_;
};

}
}
