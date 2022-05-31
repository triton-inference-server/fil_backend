#include <herring/tl_helpers.hpp>
#include <herring2/decision_forest.hpp>
#include <herring2/treelite_importer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/data_array.hpp>
#include <kayak/device_type.hpp>
#include <kayak/tree_layout.hpp>
#include <rmm/device_buffer.hpp>
#include <treelite/tree.h>
#include <treelite/gtil.h>
#include <treelite/frontend.h>
#include <xgboost/c_api.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
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

void xgb_check(int err) {
  if (err != 0) {
    throw std::runtime_error(std::string{XGBGetLastError()});
  }
}

auto load_array(std::string path, std::size_t rows, std::size_t cols) {
  auto result = std::vector<float>(rows * cols);
  auto input = std::ifstream(path, std::ifstream::binary);
  auto* buffer = reinterpret_cast<char*>(result.data());
  input.read(buffer, result.size() * sizeof(float));
  return result;
}

void run_gtil(std::unique_ptr<treelite::Model>& tl_model, matrix& input, std::vector<float>& output) {
  treelite::gtil::Predict(tl_model.get(), input.data, input.rows, output.data(), -1, true);
}

template <typename model_t>
void run_herring(model_t& final_model, matrix& input, std::vector<float>& output) {
  final_model.predict(input.data, input.rows, output.data(), 12);
}

template <typename model_t>
void run_xgb(model_t& bst, matrix& input, std::vector<float>& output) {
  auto dmat = DMatrixHandle{};
  auto const* out_result = static_cast<float*>(nullptr);
  auto out_size = bst_ulong{0};
  xgb_check(XGDMatrixCreateFromMat(
    input.data, input.rows, input.cols,
    std::numeric_limits<float>::quiet_NaN(), &dmat
  ));
  xgb_check(XGBoosterPredict(bst, dmat, 0, 0, 0, &out_size, &out_result));
  xgb_check(XGDMatrixFree(dmat));
}

void run_fil(ForestModel& model, matrix& input, float* output) {
  model.predict(output, input, true);
}

void run_herring2(
  herring::forest_model_variant& model,
  float* input,
  float* output,
  std::uint32_t rows,
  std::uint32_t cols,
  std::uint32_t out_cols
) {
  auto in = kayak::data_array<kayak::data_layout::dense_row_major, float>{
    input,
    rows,
    cols
  };
  auto out = kayak::data_array<kayak::data_layout::dense_row_major, float>{
    output,
    rows,
    out_cols
  };
  std::visit([&out, &in](auto&& concrete_model) {
    concrete_model.predict(out, in);
  }, model);
}

void run_herring2(
  herring::forest_model_variant& model,
  float* input,
  float* output,
  std::uint32_t rows,
  std::uint32_t cols,
  std::uint32_t out_cols,
  kayak::cuda_stream stream
) {
  auto in = kayak::data_array<kayak::data_layout::dense_row_major, float>{
    input,
    rows,
    cols
  };
  auto out = kayak::data_array<kayak::data_layout::dense_row_major, float>{
    output,
    rows,
    out_cols
  };
  std::visit([&out, &in, &stream](auto&& concrete_model) {
    concrete_model.predict(out, in, 0, stream);
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
  auto bst = BoosterHandle{};
  xgb_check(XGBoosterCreate(nullptr, 0, &bst));
  xgb_check(XGBoosterLoadModel(bst, model_path.c_str()));

  auto tl_model = treelite::frontend::LoadXGBoostJSONModel(model_path.c_str());
  auto converted_model = tl_model->Dispatch([](auto const& concrete_model) {
    return herring::convert_model(concrete_model);
  });
  auto final_model = std::get<2>(converted_model);
  auto herring2_model = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(*tl_model);
//#ifdef TRITON_ENABLE_GPU
  auto fil_model = ForestModel(tl_model);
  auto herring2_model_gpu = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
    *tl_model,
    128u,
    kayak::device_type::gpu,
    0,
    fil_model.get_stream()
  );
//#endif

  auto output = std::vector<float>(treelite::gtil::GetPredictOutputSize(tl_model.get(), input.rows));
  auto out_cols = output.size() / input.rows;

  auto batch_sizes = std::vector<std::size_t>{1, 16, 128, 1024, rows};
  auto batch_timings = std::vector<std::vector<std::size_t>>(4);

  // Run benchmarks for each framework
  auto start = std::chrono::high_resolution_clock::now();
  /* for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, std::min(batch, rows - j * batch), features};
      run_gtil(tl_model, cur_input, output);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }*/
  auto end = std::chrono::high_resolution_clock::now();
  /* auto gtil_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); */

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, std::min(batch, rows - j * batch), features};
      run_herring(final_model, cur_input, output);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto herring_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  for (auto i=std::size_t{}; i < std::min(output.size(), std::size_t{10}); ++i) {
    std::cout << output[i] << ", ";
    output[i] = 0;
  }
  std::cout << "\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, std::min(batch, rows - j * batch), features};
      run_herring2(herring2_model, cur_input.data, output.data() + j * batch * out_cols, cur_input.rows, cur_input.cols, out_cols);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto herring2_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  for (auto i=std::size_t{}; i < std::min(output.size(), std::size_t{10}); ++i) {
    std::cout << output[i] << ", ";
    output[i] = 0;
  }
  std::cout << "\n";

  /* start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, std::min(batch, rows - j * batch), features};
      run_xgb(bst, cur_input, output);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto xgb_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  xgb_check(XGBoosterFree(bst)); */

//  #ifdef TRITON_ENABLE_GPU
  auto gpu_buffer = rmm::device_buffer{buffer.size() * sizeof(float), fil_model.get_stream()};
  cudaMemcpy(gpu_buffer.data(), buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice);
  auto gpu_out_buffer = rmm::device_buffer{output.size() * sizeof(float), fil_model.get_stream()};

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{reinterpret_cast<float*>(gpu_buffer.data()) + j * batch, std::min(batch, rows - j * batch), features};
      run_fil(fil_model, cur_input, reinterpret_cast<float*>(gpu_out_buffer.data()));
    }
    cudaStreamSynchronize(fil_model.get_stream());
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto fil_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  cudaMemcpy(output.data(), gpu_out_buffer.data(), output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for (auto i=std::size_t{}; i < std::min(output.size(), std::size_t{10}); ++i) {
    std::cout << output[i] << ", ";
    output[i] = 0;
  }
  std::cout << "\n";

  gpu_out_buffer = rmm::device_buffer{output.size() * sizeof(float), fil_model.get_stream()};

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{reinterpret_cast<float*>(gpu_buffer.data()) + j * batch, std::min(batch, rows - j * batch), features};
      run_herring2(
        herring2_model_gpu,
        cur_input.data,
        reinterpret_cast<float*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        out_cols,
        fil_model.get_stream()
      );
    }
    cudaStreamSynchronize(fil_model.get_stream());
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto her_gpu_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
// #endif

  std::cout << "GTIL,Herring,XGBoost";
// #ifdef TRITON_ENABLE_GPU
  std::cout << ",FIL,";
// #endif
  std::cout << "\n";
  // std::cout << gtil_elapsed << ",";
  std::cout << herring_elapsed << ",";
  std::cout << herring2_elapsed << ",";
  // std::cout << xgb_elapsed << ",";
// #ifdef TRITON_ENABLE_GPU
  std::cout << fil_elapsed << ",";
  std::cout << her_gpu_elapsed << ",";
// #endif
  std::cout << "\n";
  /* std::cout << "GTIL";
  for (auto res : batch_timings[0]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "Herring";
  for (auto res : batch_timings[1]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "XGBoost";
  for (auto res : batch_timings[2]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "FIL";
  for (auto res : batch_timings[3]) {
    std::cout << "," << res;
  }
  std::cout << "\n"; */

  return 0;
}
