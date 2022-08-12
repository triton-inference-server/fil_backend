#include <herring3/decision_forest.hpp>
#include <herring3/predict.hpp>
#include <herring3/treelite_importer.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/data_array.hpp>
#include <kayak/device_type.hpp>
#include <kayak/tree_layout.hpp>
#include <nvtx3/nvtx3.hpp>
#include <rmm/device_buffer.hpp>
#include <treelite/tree.h>
#include <treelite/gtil.h>
#include <treelite/frontend.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <iostream>
#include <memory>
#include <limits>
#include <chrono>
#include <algorithm>

#include <matrix.hpp>
//#ifdef TRITON_ENABLE_GPU
#include <run_fil.hpp>
#include <cuda_runtime_api.h>
//#endif

auto load_array(std::string path, std::size_t rows, std::size_t cols) {
  auto result = std::vector<float>(rows * cols);
  auto input = std::ifstream(path, std::ifstream::binary);
  auto* buffer = reinterpret_cast<char*>(result.data());
  input.read(buffer, result.size() * sizeof(float));
  return result;
}

void run_fil(ForestModel& model, matrix& input, float* output) {
  model.predict(output, input, true);
}

void run_herring3(
  herring::forest_model_variant& model,
  float* input,
  float* output,
  std::size_t rows,
  std::size_t cols,
  kayak::cuda_stream stream
) {
  // NVTX3_FUNC_RANGE();
  std::visit([output, input, rows, cols, &stream](auto&& concrete_model) {
    predict(concrete_model.obj(), concrete_model.get_postprocessor(), output, input, rows, cols, concrete_model.num_class(), std::nullopt, 0, stream);
  }, model);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " [XGBoost model path] [Data path] [rows] [features]";
    return 1;
  }

  auto model_path = std::string{argv[1]};
  auto data_path = std::string{argv[2]};
  auto rows = static_cast<std::size_t>(std::stol(std::string{argv[3]}));
  auto features = static_cast<std::size_t>(std::stol(std::string{argv[4]}));

  auto buffer = load_array(data_path, rows, features);
  auto input = matrix{buffer.data(), rows, features};

  auto tl_model = treelite::frontend::LoadXGBoostJSONModel(model_path.c_str());

  auto fil_model = ForestModel(tl_model);
  herring::initialize_gpu_options();
  auto herring3_model_gpu = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
    *tl_model,
    128u,
    kayak::device_type::gpu,
    0,
    fil_model.get_stream()
  );

  auto output = std::vector<float>(treelite::gtil::GetPredictOutputSize(tl_model.get(), input.rows));
  auto out_cols = output.size() / input.rows;

  auto half_index = output.size() / 2;

  // auto batch_sizes = std::vector<std::size_t>{1, 16, 128, 1024, rows};
  // auto batch_sizes = std::vector<std::size_t>{1, 2, 4, 8, 16};
  auto batch_sizes = std::vector<std::size_t>{2};
  auto batch_timings = std::vector<std::vector<std::size_t>>(4);

  // Run benchmarks for each framework
  auto fil_output = std::vector<float>(2 * output.size());
  auto gpu_buffer = rmm::device_buffer{buffer.size() * sizeof(float), fil_model.get_stream()};
  cudaMemcpy(gpu_buffer.data(), buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice);
  auto gpu_out_buffer = rmm::device_buffer{fil_output.size() * sizeof(float), fil_model.get_stream()};

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{reinterpret_cast<float*>(gpu_buffer.data()) + j * batch, std::min(batch, rows - j * batch), features};
      run_fil(fil_model, cur_input, reinterpret_cast<float*>(gpu_out_buffer.data()));
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto fil_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  cudaMemcpy(fil_output.data(), gpu_out_buffer.data(), fil_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "WH: " << fil_output.data()[1] << ", " << fil_output.data()[half_index * 2 + 1] << ", " << fil_output.data()[output.size() * 2 - 1] << "\n";

  start = std::chrono::high_resolution_clock::now();

  gpu_out_buffer = rmm::device_buffer{output.size() * sizeof(float), fil_model.get_stream()};

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{reinterpret_cast<float*>(gpu_buffer.data()) + j * batch, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<float*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream()
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto her_gpu_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  auto her_output = std::vector<float>(output.size());
  cudaMemcpy(her_output.data(), gpu_out_buffer.data(), her_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "WH: " << her_output.data()[0] << ", " << her_output.data()[half_index] << ", " << her_output.data()[output.size() - 1] << "\n";

  std::cout << "FIL, Herring3" << "\n";
  std::cout << fil_elapsed << ", " << her_gpu_elapsed << "\n";
  std::cout << "Framework";
  for (auto size : batch_sizes) {
    std::cout << "," << size;
  }
  std::cout << "\n";
  std::cout << "FIL";
  for (auto res : batch_timings[0]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "Herring3-GPU";
  for (auto res : batch_timings[1]) {
    std::cout << "," << res;
  }
  std::cout << "\n";

  return 0;
}
