#pragma once
#include <cmath>
#include <cstddef>
#include <stdint.h>
#include <numeric>
#include <optional>
#include <vector>
#include <herring/output_ops.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>
#include <kayak/detail/index_type.hpp>

namespace herring {

struct model_builder_error : std::exception {
  model_builder_error() : model_builder_error("Error while building model") {}
  model_builder_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template<typename decision_forest_t>
struct decision_forest_builder {
  using index_type = typename decision_forest_t::index_type;
  using raw_index_t = kayak::raw_index_t;
  using offset_type = typename decision_forest_t::offset_type;
  using feature_index_type = typename decision_forest_t::feature_index_type;
  using node_value_type  = typename decision_forest_t::node_value_type;
  using output_index_type  = typename decision_forest_t::output_index_type;
  using threshold_type = typename node_value_type::value_type;
  using output_type = typename decision_forest_t::output_type;
  using category_set_type = typename decision_forest_t::category_set_type;

  void start_new_tree() {
    if (tree_offsets_.size() == 0) {
      tree_offsets_.emplace_back();
    } else {
      if (alignment_ != 0) {
        if (cur_tree_size_ % alignment_ != 0) {
          auto padding = (alignment_ - cur_tree_size_ % alignment_);
          for (auto i = std::size_t{}; i < padding; ++i) {
            add_empty_node();
          }
        }
      }
      tree_offsets_.push_back(
        tree_offsets_.back() + cur_tree_size_
      );
    }
    cur_tree_size_ = 0;
  }

  void add_threshold_node(offset_type distant_child_offset, feature_index_type fea, threshold_type threshold, bool default_distant, bool is_inclusive=false) {
    if (is_inclusive) {
      threshold = std::nextafter(threshold, std::numeric_limits<threshold_type>::infinity());
    }
    node_values_.push_back({.value=threshold});
    add_node(distant_child_offset, fea, default_distant);
  }

  void add_leaf_node(threshold_type output_val) {
    node_values_.push_back({.value=output_val});
    add_node();
  }

  void add_leaf_node(output_index_type output_val) {
    node_values_.push_back({.index=output_val});
    add_node();
  }


  template<
    typename output_t=output_type,
    typename=std::enable_if_t<!std::is_same_v<threshold_type, output_t> && !std::is_same_v<output_index_type, output_t>>
  >
  void add_leaf_node(output_t output_val) {
    if (!node_outputs_) {
      node_outputs_ = std::vector<output_type>{};
    }
    node_values_.push_back({.index=node_outputs_->size()});
    node_outputs_->push_back(output_val);
    add_node();
  }

  template<typename iter_t>
  void add_leaf_node(iter_t output_begin, iter_t output_end) {
    auto num_values = std::distance(output_begin, output_end);
    if (output_size_) {
      if (*output_size_ != num_values) {
        throw model_builder_error("Inconsistent vector leaf sizes");
      }
    } else {
      output_size_ = num_values;
    }
    if (output_size_ == 1 && (std::is_same_v<threshold_type, output_type> || std::is_same_v<output_index_type, output_type>)) {
      add_leaf_node(*output_begin);
    } else {
      if (!node_outputs_) {
        node_outputs_ = std::vector<output_type>{};
      }
      node_values_.push_back({.index=node_outputs_->size()});
      std::copy(output_begin, output_end, std::back_inserter(*node_outputs_));
      add_node();
    }
  }

  template<typename iter_t>
  void add_categorical_node(offset_type distant_child_offset, feature_index_type fea, iter_t begin_categories, iter_t end_categories, bool default_distant) {
    auto max_category = std::max_element(begin_categories, end_categories) + 1;
    if constexpr (decision_forest_t::categorical_lookup) {
      if (!categorical_sizes_) {
        categorical_storage_ = std::vector<uint8_t>{};
      }
      node_values_.push_back({.index=categorical_storage_->size()});

      auto bins = (
        max_category / category_set_type::bin_width +
        bool{max_category % category_set_type::bin_width != 0}
      );
      for (auto i = std::size_t{}; i < bins; ++i) {
        categorical_storage_->emplace_back();
      }

      auto category_bitset = category_set_type{
        &((*categorical_storage_)[node_values_.back().index]),
        max_category
      };
      std::for_each(begin_categories, end_categories, [](auto&& cat) {
        category_bitset.set(cat);
      });
    } else {
      node_values_.push_back({.index=output_index_type{}});
      auto category_bitset = category_set_type{&node_values_.back(), max_category};
      if (max_category > category_bitset.bin_width) {
        throw model_builder_error("Too many categories for non-lookup categorical node");
      }
      std::for_each(begin_categories, end_categories, [](auto&& cat) {
        category_bitset.set(cat);
      });
    }
    add_node(distant_child_offset, fea, default_distant);
  }

  void add_empty_node() {
    node_values_.push_back({.value=threshold_type{}});
    add_node();
  }

  void set_element_postproc(element_op val) { element_postproc_ = val; }
  void set_row_postproc(row_op val) { row_postproc_ = val; }
  void set_average_factor(double val) { average_factor_ = val; }
  void set_bias(double val) { bias_ = val; }
  void set_postproc_constant(double val) { postproc_constant_ = val; }

  decision_forest_builder(index_type align_bytes=index_type{}) :
    cur_tree_size_{},
    alignment_{std::lcm(align_bytes.value(), align_unit)},
    output_size_{},
    element_postproc_{},
    row_postproc_{},
    average_factor_{},
    bias_{},
    postproc_constant_{},
    tree_offsets_{},
    node_storage_size{},
    node_values_{},
    node_features_{},
    node_offsets_{},
    default_distant_{},
    node_outputs_{},
    categorical_sizes_{},
    categorical_storage_{} {
  }

  auto get_decision_forest(
      index_type num_class,
      index_type num_features,
      kayak::device_type mem_type=kayak::device_type::cpu,
      int device=0,
      kayak::cuda_stream stream=kayak::cuda_stream{}
  ) {
    auto tree_offsets_buf = kayak::buffer{tree_offsets_.data(), tree_offsets_.size()};
    auto node_values_buf = kayak::buffer{node_values_.data(), node_values_.size()};
    auto node_features_buf = kayak::buffer{node_features_.data(), node_features_.size()};
    auto node_offsets_buf = kayak::buffer{node_offsets_.data(), node_offsets_.size()};
    auto default_distant_buf = kayak::buffer<bool>{
      std::begin(default_distant_),
      std::end(default_distant_),
      mem_type,
      device,
      stream
    };
    auto node_outputs_buf = std::optional<kayak::buffer<output_type>>{};
    if (node_outputs_) {
      node_outputs_buf = kayak::buffer{
        kayak::buffer{node_outputs_->data(), node_outputs_->size()},
        mem_type,
        device,
        stream
      };
    }
    auto categorical_sizes_buf = std::optional<kayak::buffer<raw_index_t>>{};
    if (categorical_sizes_) {
      categorical_sizes_buf = kayak::buffer{
        kayak::buffer{categorical_sizes_->data(), categorical_sizes_->size()},
        mem_type,
        device,
        stream
      };
    }
    auto categorical_storage_buf = std::optional<kayak::buffer<uint8_t>>{};
    if (categorical_storage_) {
      categorical_storage_buf = kayak::buffer{
        kayak::buffer{categorical_storage_->data(), categorical_storage_->size()},
        mem_type,
        device,
        stream
      };
    }

    return decision_forest_t{
      num_class,
      num_features,
      kayak::buffer{tree_offsets_buf, mem_type, device, stream},
      kayak::buffer{node_values_buf, mem_type, device, stream},
      kayak::buffer{node_features_buf, mem_type, device, stream},
      kayak::buffer{node_offsets_buf, mem_type, device, stream},
      std::move(default_distant_buf),
      std::move(node_outputs_buf),
      std::move(categorical_sizes_buf),
      std::move(categorical_storage_buf),
      element_postproc_,
      row_postproc_,
      average_factor_,
      bias_,
      postproc_constant_,
      output_size_.value_or(1u)
    };
  }


 private:
  std::size_t cur_tree_size_;
  raw_index_t alignment_;
  std::optional<raw_index_t> output_size_;
  element_op element_postproc_;
  row_op row_postproc_;
  double average_factor_;
  double bias_;
  double postproc_constant_;

  std::vector<raw_index_t> tree_offsets_;
  raw_index_t node_storage_size;
  std::vector<node_value_type> node_values_;
  std::vector<feature_index_type> node_features_;
  std::vector<offset_type> node_offsets_;
  std::vector<bool> default_distant_;

  std::optional<std::vector<output_type>> node_outputs_;
  std::optional<std::vector<raw_index_t>> categorical_sizes_;
  std::optional<std::vector<uint8_t>> categorical_storage_;

  auto constexpr static const align_unit = std::lcm(
    sizeof(raw_index_t),
    std::lcm(
      sizeof(node_value_type),
      std::lcm(
        sizeof(feature_index_type),
        sizeof(offset_type)
      )
    )
  );

  void add_node(offset_type offset=offset_type{}, feature_index_type fea=feature_index_type{}, bool default_distant=false) {
    node_offsets_.push_back(offset);
    node_features_.push_back(fea);
    default_distant_.push_back(default_distant);
    ++cur_tree_size_;
  }
};

}
