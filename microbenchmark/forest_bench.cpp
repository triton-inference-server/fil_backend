#include <herring/tl_helpers.hpp>
#include <treelite/tree.h>
#include <treelite/gtil.h>
#include <treelite/frontend.h>
// #include <xgboost/c_api.h>

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

/* void xgb_check(int err) {
  if (err != 0) {
    throw std::runtime_error(std::string{XGBGetLastError()});
  }
} */

struct matrix {
  float* data;
  std::size_t rows;
  std::size_t cols;
};

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
  final_model.predict(input.data, input.rows, output.data());
}

/* template <typename model_t>
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
} */

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
  /* auto bst = BoosterHandle{};
  xgb_check(XGBoosterCreate(nullptr, 0, &bst));
  xgb_check(XGBoosterLoadModel(bst, model_path.c_str())); */

  auto tl_model = treelite::frontend::LoadXGBoostJSONModel(model_path.c_str());
  auto converted_model = tl_model->Dispatch([](auto const& concrete_model) {
    return herring::convert_model(concrete_model);
  });
  auto final_model = std::get<2>(converted_model);

  auto output = std::vector<float>(treelite::gtil::GetPredictOutputSize(tl_model.get(), input.rows));

  auto batch_sizes = std::vector<std::size_t>{1024};

  // Run benchmarks for each framework
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch;

    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, batch, features};
      run_gtil(tl_model, cur_input, output);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto gtil_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch;

    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, batch, features};
      run_herring(final_model, cur_input, output);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto herring_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  /* start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch;

    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix{buffer.data() + j * batch, batch, features};
      run_xgb(bst, cur_input, output);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto xgb_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  xgb_check(XGBoosterFree(bst)); */

  std::cout << "GTIL,Herring,XGBoost\n";
  std::cout << gtil_elapsed << ",";
  std::cout << herring_elapsed << ",";
  // std::cout << xgb_elapsed << "\n";
  std::cout << "\n";

  return 0;
}
