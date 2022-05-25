#pragma once
#include <kayak/buffer.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/device_type.hpp>
#include <kayak/ndarray.hpp>
#include <type_traits>

namespace herring {
namespace detail {
namespace inference {

using kayak::raw_index_t;

namespace cpu {
#ifdef __cpp_lib_hardware_interference_size
  using std::hardware_constructive_interference_size;
#else
  auto constexpr hardware_constructive_interference_size = std::size_t{64};
#endif

auto constexpr static const CHUNK_SIZE = hardware_constructive_interference_size;
auto constexpr static const GROVE_SIZE = hardware_constructive_interference_size;

}

template<
  kayak::device_type D,
  typename forest_t,
  typename io_t
>
std::enable_if_t<D == kayak::device_type::cpu, void> predict(
  forest_t const& forest, 
  kayak::data_array<kayak::data_layout::dense_row_major, io_t>& out,
  kayak::data_array<kayak::data_layout::dense_row_major, io_t> const& in,
  raw_index_t num_class,
  raw_index_t leaf_size
) {
  auto const num_trees = forest.tree_count();
  auto const num_rows = in.rows();
  auto const num_chunks = num_rows / cpu::CHUNK_SIZE + (num_rows % cpu::CHUNK_SIZE != 0);
  auto const num_groves = num_trees / cpu::GROVE_SIZE + (num_trees % cpu::GROVE_SIZE != 0);
  auto const num_tasks = num_chunks * num_groves;

  auto workspace_vec = std::vector<typename forest_t::output_type>(
    num_rows * num_class * num_groves
  );
  auto workspace_buf = kayak::buffer<typename forest_t::output_type>(
    workspace_vec.data(), num_rows * num_class * num_groves
  );
  // PERF (whicks): Try 0, 2, 1 layout
  auto workspace = kayak::ndarray<typename forest_t::output_t, 0, 1, 2>(workspace_buf.data(), num_rows, num_class, num_groves);

  auto missing_values_buf = kayak::buffer<typename forest_t::output_type>(in.size());
  auto missing_values = kayak::data_array<decltype(in)::layout, bool>(
      missing_values_buf.data(),
      in.rows(),
      in.cols()
  );

#pragma omp parallel
  for(auto task_index = std::size_t{}; task_index < num_tasks; ++task_index) {
    auto const grove_index = task_index / num_chunks;
    auto const chunk_index = task_index % num_chunks;
    auto const row_start = chunk_index * cpu::CHUNK_SIZE;
    auto const row_end = std::min(row_start + cpu::CHUNK_SIZE, num_rows);
    auto const tree_start = grove_index * cpu::GROVE_SIZE;
    auto const tree_end = std::min(tree_start + cpu::GROVE_SIZE, num_trees);

    for (auto row_index = row_start; row_index < row_end; ++row_index) {
      for (auto tree_index = tree_start; tree_index < tree_end; ++tree_index) {
        auto tree_out = forest.evaluate_tree(tree_index, row_index, in);
        if constexpr (!vector_leaf) {
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
}

template<
  kayak::device_type D,
  typename forest_t,
  typename out_array_t,
  typename in_array_t
>
std::enable_if_t<D == kayak::device_type::gpu && !kayak::GPU_ENABLED, void> predict(
  forest_t const& forest, 
  out_array_t& out,
  in_array_t const& in,
  raw_index_t num_class,
  raw_index_t leaf_size
) {
  throw kayak::gpu_unsupported(
    "Attempting to launch forest inference on device in non-GPU build"
  );
}

}
}
}
