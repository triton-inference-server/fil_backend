#pragma once

#include <iostream>
#include <memory>
#include <algorithm_val.hpp>
#include <results.hpp>
#include <cuml/fil/fil.h>
#include <input_data.hpp>
#include <kayak/buffer.hpp>
#include <kayak/ceildiv.hpp>
#include <kayak/device_type.hpp>
#include <treelite/gtil.h>
#include <treelite/tree.h>
#include <raft/handle.hpp>

template <algorithm_val A, typename T>
auto run_fil(
  std::unique_ptr<treelite::Model> const& tl_model,
  input_data<T> const& input,
  std::vector<std::size_t> const& batch_sizes
) {
  auto handle = raft::handle_t{};
  auto model = ML::fil::forest_variant{};
  auto fil_algo = ML::fil::algo_t::ALGO_AUTO;
  auto fil_storage = ML::fil::storage_type_t::AUTO;
  auto fil_precision = ML::fil::precision_t::PRECISION_NATIVE;
  if constexpr (std::is_same_v<T, float>) {
    fil_precision = ML::fil::precision_t::PRECISION_FLOAT32;
  } else {
    fil_precision = ML::fil::precision_t::PRECISION_FLOAT64;
  }
  if constexpr (A == algorithm_val::fil_sparse || A == algorithm_val::fil_sparse8) {
    fil_algo = ML::fil::algo_t::NAIVE;
    fil_storage = ML::fil::storage_type_t::SPARSE;
  } else if constexpr (A == algorithm_val::fil_dense) {
    fil_algo = ML::fil::algo_t::BATCH_TREE_REORG;
    fil_storage = ML::fil::storage_type_t::DENSE;
  } else if constexpr (A == algorithm_val::fil_dense_reorg) {
    fil_algo = ML::fil::algo_t::TREE_REORG;
    fil_storage = ML::fil::storage_type_t::DENSE;
  }

  auto in_buffer = kayak::buffer<T>{
    input.data,
    kayak::device_type::gpu,
    0,
    handle.get_stream().value()
  };
  auto out_buffer = kayak::buffer<T>{
    treelite::gtil::GetPredictOutputSize(tl_model.get(), input.rows) * 2,
    kayak::device_type::gpu,
    0,
    handle.get_stream().value()
  };
  auto threads_per_tree_vals = std::vector<int>{1, 2, 4, 8, 16, 32};

  auto result = std::vector<benchmark_results>{};

  for (auto tpt : threads_per_tree_vals) {
    auto config = ML::fil::treelite_params_t {
      fil_algo,
      true,
      0.5,
      fil_storage,
      0,
      tpt,
      0,
      nullptr,
      fil_precision
    };
    ML::fil::from_treelite(
      handle,
      &model,
      static_cast<void*>(tl_model.get()),
      &config
    );
    auto label = std::string{"FIL"};
    if constexpr (A == algorithm_val::fil_sparse) {
      label += std::string{"-SPA-"};
    } else if constexpr (A == algorithm_val::fil_sparse8) {
      label += std::string{"-SP8-"};
    } else if constexpr (A == algorithm_val::fil_dense) {
      label += std::string{"-DBR-"};
    } else if constexpr (A == algorithm_val::fil_dense_reorg) {
      label += std::string{"-DTR-"};
    }
    label += std::to_string(tpt);

    result.emplace_back(label, std::vector<std::size_t>{});
    auto& chunk_result = result.back();

    for (auto batch_size : batch_sizes) {
      auto total_batches = kayak::ceildiv(input.rows, batch_size);
      auto start = std::chrono::high_resolution_clock::now();
      for (auto batch_index = std::size_t{}; batch_index < total_batches; ++batch_index) {
        auto cur_rows = std::min(batch_size, input.rows - batch_index * batch_size);
        auto batch_in = kayak::buffer<T>{
          in_buffer.data() + batch_index * batch_size * input.cols,
          cur_rows * input.cols,
          kayak::device_type::gpu,
          0
        };
        std::visit([&handle, &out_buffer, &batch_in, cur_rows](auto&& forest) {
          if constexpr (
            std::is_same_v<std::remove_reference_t<decltype(forest)>, ML::fil::forest_t<T>>
          ) {
            ML::fil::predict(
              handle,
              forest,
              out_buffer.data(),
              batch_in.data(),
              cur_rows,
              true
            );
          }
        }, model);
      }
      handle.sync_stream();

      auto end = std::chrono::high_resolution_clock::now();
      chunk_result.elapsed_times.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
      );
    }
    std::visit([&handle](auto&& forest) { ML::fil::free(handle, forest); }, model);
  }
  auto print_buffer = kayak::buffer{
    out_buffer,
    kayak::device_type::cpu
  };
  for (auto i = std::size_t{}; i < 12; ++i) {
    std::cout << print_buffer.data()[i * 2 + 1] << ", ";
  }
  std::cout << "\n";
  return result;
}
