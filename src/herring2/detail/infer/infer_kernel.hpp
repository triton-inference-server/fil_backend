#pragma once
#include <cmath>
#include <herring/output_ops.hpp>
#include <herring2/detail/cpu_constants.hpp>
#include <herring2/detail/postprocess.hpp>
#include <kayak/data_array.hpp>
#include <kayak/flat_array.hpp>
#include <kayak/ndarray.hpp>

namespace herring {
namespace detail {
namespace inference {
namespace cpu{

using kayak::raw_index_t;

template <
  bool categorical,
  bool lookup,
  bool vector_leaf,
  typename forest_t,
  typename io_t
>
void infer_kernel(
    forest_t forest,
    kayak::data_array<kayak::data_layout::dense_row_major, io_t> out,
    kayak::data_array<kayak::data_layout::dense_row_major, io_t> in,
    raw_index_t num_class,
    element_op element_postproc,
    row_op row_postproc,
    io_t average_factor,
    io_t bias,
    io_t postproc_constant
) {
  auto const num_trees = forest.tree_count();
  auto const num_rows = in.rows();
  auto const num_chunks = num_rows / CHUNK_SIZE + (num_rows % CHUNK_SIZE != 0);
  auto const num_groves = num_trees / GROVE_SIZE + (num_trees % GROVE_SIZE != 0);
  auto const num_tasks = num_chunks * num_groves;

  auto workspace_vec = std::vector<io_t>(
    num_rows * num_class * num_groves
  );
  auto workspace_buf = kayak::buffer<io_t>(
    workspace_vec.data(), num_rows * num_class * num_groves
  );
  auto workspace = kayak::ndarray<io_t, 2, 0, 1>(workspace_buf.data(), num_rows, num_class, num_groves);

  auto missing_values_buf = kayak::buffer<bool>(in.size());
  auto missing_values = kayak::data_array<decltype(in)::layout, bool>(
      missing_values_buf.data(),
      in.rows(),
      in.cols()
  );
  auto has_missing = false;
  for (auto row_index = raw_index_t{}; row_index < in.rows(); ++row_index) {
    for (auto col_index = raw_index_t{}; col_index < in.cols(); ++col_index) {
      auto nan_value = std::isnan(in.at(row_index, col_index));
      missing_values.at(row_index, col_index) = nan_value;
      has_missing = has_missing || nan_value;
    }
  }

#pragma omp parallel
  for(auto task_index = raw_index_t{}; task_index < num_tasks; ++task_index) {
    auto const grove_index = task_index / num_chunks;
    auto const chunk_index = task_index % num_chunks;
    auto const row_start = chunk_index * CHUNK_SIZE;
    auto const row_end = std::min(row_start + CHUNK_SIZE, num_rows);
    auto const tree_start = grove_index * GROVE_SIZE;
    auto const tree_end = std::min(tree_start + GROVE_SIZE, num_trees);

    for (auto row_index = row_start; row_index < row_end; ++row_index) {
      for (auto tree_index = tree_start; tree_index < tree_end; ++tree_index) {
        auto tree_out = kayak::flat_array<kayak::array_encoding::dense, typename forest_t::output_type const>{};
        if (!has_missing) {
          tree_out = forest.template evaluate_tree<categorical, true, lookup>(tree_index, row_index, in);
        } else {
          tree_out = forest.template evaluate_tree<categorical, lookup>(tree_index, row_index, in, missing_values);
        }
        if (tree_out.size() == 0) {
          auto class_index = tree_index % num_class;
          workspace.at(row_index, class_index, grove_index) += tree_out.at(0);
        } else {
          for (auto class_index = raw_index_t{}; class_index < num_class; ++class_index) {
            workspace.at(row_index, class_index, grove_index) += tree_out.at(class_index);
          }
        }
      }
    }
  }

#pragma omp parallel
  for (auto chunk_index = raw_index_t{}; chunk_index < num_chunks; ++chunk_index) {
    for (auto row_index = chunk_index * CHUNK_SIZE; row_index < in.rows() && row_index < (chunk_index + 1) * CHUNK_SIZE; ++row_index) {
      auto final_output = postprocess(
        workspace,
        row_index,
        element_postproc,
        row_postproc,
        average_factor,
        bias,
        postproc_constant
      );
      for (auto i = raw_index_t{}; i < final_output.size(); ++i) {
        out.at(row_index, i) = final_output.at(i);
      }
    }
  }
}

}
}
}
}
