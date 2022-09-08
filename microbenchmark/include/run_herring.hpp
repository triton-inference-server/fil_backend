#pragma once
#include <chrono>
#include <cstddef>
#include <vector>
#include <results.hpp>
#include <herring3/forest_model.hpp>
#include <input_data.hpp>
#include <kayak/buffer.hpp>
#include <kayak/ceildiv.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>
#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

template <kayak::device_type D, typename T>
auto run_herring(
  herring::forest_model& model,
  input_data<T> const& input,
  std::vector<std::size_t> const& batch_sizes
) {
  auto in_buffer = kayak::buffer<T>{input.data.data(), input.data.size()};
  auto stream = kayak::cuda_stream{};
  if constexpr (D == kayak::device_type::gpu) {
    in_buffer = kayak::buffer<T>{input.data, D, 0, stream};
  }
  auto out_buffer = kayak::buffer<T>{
    input.rows * model.num_outputs(), D, 0, stream
  };
  auto chunk_sizes = std::vector<std::size_t>{2, 4, 8, 16, 32};
  if constexpr (D == kayak::device_type::gpu) {
    chunk_sizes = std::vector<std::size_t>{8};
  } else {
    chunk_sizes = std::vector<std::size_t>{64};
  }

  auto result = std::vector<benchmark_results>{};

  for (auto chunk_size : chunk_sizes) {
    auto label = std::string{"HER"};
    if constexpr (D == kayak::device_type::gpu) {
      label += std::string{"-GPU-"};
    } else {
      label += std::string{"-CPU-"};
    }
    label += std::to_string(chunk_size);

    result.emplace_back(label, std::vector<std::size_t>{});
    auto& chunk_result = result.back();

    for (auto batch_size : batch_sizes) {
      auto total_batches = kayak::ceildiv(input.rows, batch_size);
      auto start = std::chrono::high_resolution_clock::now();
      for (auto batch_index = std::size_t{}; batch_index < total_batches; ++batch_index) {
        auto batch_in = kayak::buffer<T>{
          in_buffer.data() + batch_index * batch_size * input.cols,
          std::min(batch_size, input.rows - batch_index * batch_size) * input.cols,
          D,
          0
        };
        model.predict(out_buffer, batch_in, stream, chunk_size);
      }

      if constexpr (D == kayak::device_type::gpu) {
#ifdef ENABLE_GPU
        kayak::cuda_check(cudaStreamSynchronize(stream));
#endif
      }

      auto end = std::chrono::high_resolution_clock::now();
      chunk_result.elapsed_times.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
      );
    }
  }
  auto print_buffer = kayak::buffer{
    out_buffer,
    kayak::device_type::cpu
  };
  for (auto i = std::size_t{}; i < 12; ++i) {
    std::cout << print_buffer.data()[i] << ", ";
  }
  std::cout << "\n";
  return result;
}
