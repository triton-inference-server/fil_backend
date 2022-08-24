#pragma once
#include <cstddef>
#include <iostream>
#include <new>
#include <numeric>
#include <vector>
#include <herring3/detail/cpu_introspection.hpp>
#include <herring3/detail/evaluate_tree.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <kayak/ceildiv.hpp>

namespace herring {
namespace detail {

template<bool is_categorical, typename forest_t,
  typename vector_output_t=std::nullptr_t>
void infer_kernel_cpu(
    forest_t const& forest,
    postprocessor<
      typename forest_t::leaf_output_type, typename forest_t::io_type
    > const& postproc,
    typename forest_t::io_type* output,
    typename forest_t::io_type const* input,
    std::size_t row_count,
    std::size_t col_count,
    std::size_t num_outputs,
    std::size_t chunk_size=hardware_constructive_interference_size,
    std::size_t grove_size=hardware_constructive_interference_size,
    vector_output_t vector_output_p=nullptr
) {
  auto constexpr has_vector_leaves = (
    !std::is_same_v<vector_output_t, std::nullptr_t>
    && std::is_integral_v<typename forest_t::leaf_output_type>
  );
  using node_t = typename forest_t::node_type;

  using output_t = std::conditional_t<
    has_vector_leaves,
    vector_output_t,
    typename node_t::threshold_type
  >;

  auto const num_tree = forest.tree_count();
  auto const num_grove = kayak::ceildiv(num_tree, grove_size);
  auto const num_chunk = kayak::ceildiv(row_count, chunk_size);

  auto output_workspace = std::vector<output_t>(
    row_count * num_outputs * num_grove,
    output_t{}
  );
  auto const task_count = num_grove * num_chunk;

  // Infer on each grove and chunk
#pragma omp parallel for
  for(auto task_index = std::size_t{}; task_index < task_count; ++task_index) {
    auto const grove_index = task_index / num_chunk;
    auto const chunk_index = task_index % num_chunk;
    auto const start_row = chunk_index * chunk_size;
    auto const end_row = std::min(start_row + chunk_size, row_count);
    auto const start_tree = grove_index * grove_size;
    auto const end_tree = std::min(start_tree + grove_size, num_tree);

    for (auto row_index = start_row; row_index < end_row; ++row_index){
      for (auto tree_index = start_tree; tree_index < end_tree; ++tree_index) {
        if constexpr (has_vector_leaves) {
          auto leaf_index = evaluate_tree<typename
            forest_t::leaf_output_type
          >(
            forest.get_tree_root(tree_index),
            input + row_index * col_count
          );
          for (
            auto class_index=std::size_t{};
            class_index < num_outputs;
            ++class_index
          ) {
            output_workspace[
              row_index * num_outputs * num_grove
              + class_index * num_grove
              + grove_index
            ] += vector_output_p[
              leaf_index * num_outputs + class_index
            ];
          }
        } else {
          auto output_offset = (
            row_index * num_outputs * num_grove
            + (tree_index % num_outputs) * num_grove
            + grove_index
          );
          // std::cout << output_offset << "\n";
          output_workspace[output_offset] += evaluate_tree<
            typename forest_t::leaf_output_type
          >(
            forest.get_tree_root(tree_index),
            input + row_index * col_count
          );
        }
      }  // Trees
    }  // Rows
  }  // Tasks

  // Sum over grove and postprocess
#pragma omp parallel for
  for (auto row_index=std::size_t{}; row_index < row_count; ++row_index) {
    for (
      auto class_index = std::size_t{};
      class_index < num_outputs;
      ++class_index
    ) {
      auto grove_offset = (
        row_index * num_outputs * num_grove + class_index * num_grove
      );

      output_workspace[grove_offset] = std::accumulate(
        std::begin(output_workspace) + grove_offset,
        std::begin(output_workspace) + grove_offset + num_grove,
        output_t{}
      );
    }
    postproc(
      output_workspace.data() + row_index * num_outputs * num_grove,
      num_outputs,
      output + row_index * num_outputs
    );
  }
}

}
}
