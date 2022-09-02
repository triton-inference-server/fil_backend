#include <herring/tl_helpers.hpp>
#include <herring3/decision_forest.hpp>
#include <herring3/treelite_importer.hpp>
#include <kayak/buffer.hpp>
#include <kayak/cuda_check.hpp>
#include <kayak/cuda_stream.hpp>
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


using real_t=float;

auto load_array(std::string path, std::size_t rows, std::size_t cols) {
  auto result = std::vector<float>(rows * cols);
  auto input = std::ifstream(path, std::ifstream::binary);
  auto* buffer = reinterpret_cast<char*>(result.data());
  input.read(buffer, result.size() * sizeof(float));
  if constexpr (!std::is_same_v<real_t, float>) {
    auto copied_result = std::vector<real_t>(rows * cols);
    std::copy(std::begin(result), std::end(result), std::begin(copied_result));
    return copied_result;
  } else {
    return result;
  }
}

/* void run_fil(ForestModel& model, matrix<real_t>& input, real_t* output) {
  model.predict(output, input, true);
} */

template <typename model_t>
void run_old_herring(model_t& final_model, matrix<real_t>& input, std::vector<real_t>& output) {
  final_model.predict(input.data, input.rows, output.data(), 12);
}

void run_gtil(std::unique_ptr<treelite::Model>& tl_model, matrix<real_t>& input, std::vector<real_t>& output) {
  // treelite::gtil::Predict(tl_model.get(), input.data, input.rows, output.data(), -1, true);
}

void run_herring3(
  herring::forest_model& model,
  real_t* input,
  real_t* output,
  std::size_t rows,
  std::size_t cols,
  kayak::cuda_stream stream,
  std::size_t rpbi
) {
  auto in_buf = kayak::buffer(
    input,
    rows * cols,
    kayak::device_type::gpu
  );
  auto out_buf = kayak::buffer(
    output,
    rows * model.num_outputs(),
    kayak::device_type::gpu
  );
  model.predict(out_buf, in_buf, stream, rpbi);
}

void run_herring3_cpu(
  herring::forest_model& model,
  real_t* input,
  real_t* output,
  std::size_t rows,
  std::size_t cols,
  kayak::cuda_stream stream,
  std::size_t rpbi
) {
  auto in_buf = kayak::buffer(
    input,
    rows * cols,
    kayak::device_type::cpu
  );
  auto out_buf = kayak::buffer(
    output,
    rows * model.num_outputs(),
    kayak::device_type::cpu
  );
  model.predict(out_buf, in_buf, stream, rpbi);
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
  auto input = matrix<real_t>{buffer.data(), rows, features};

  auto tl_model = treelite::frontend::LoadXGBoostJSONModel(model_path.c_str());

  /* auto old_converted_model = tl_model->Dispatch([](auto const& concrete_model) {
    return herring_old::convert_model(concrete_model);
  });
  auto old_herring_model = std::get<2>(old_converted_model); */

  auto fil_model = ForestModel(tl_model, false);
  auto fil_model_sparse = ForestModel(tl_model, true);
  auto herring3_model_gpu = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
    *tl_model,
    128u,
    std::is_same_v<real_t, double>,
    kayak::device_type::gpu,
    0,
    fil_model.get_stream()
  );
  auto herring3_model_cpu = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
    *tl_model,
    128u,
    std::is_same_v<real_t, double>,
    kayak::device_type::cpu
  );

  auto output = std::vector<real_t>(treelite::gtil::GetPredictOutputSize(tl_model.get(), input.rows));
  auto out_cols = output.size() / input.rows;

  auto half_index = output.size() / 2;

  // auto batch_sizes = std::vector<std::size_t>{1, 16, 128, 1024, rows};
  // auto batch_sizes = std::vector<std::size_t>{1, 2, 4, 8, 16};
  // auto batch_sizes = std::vector<std::size_t>{1};
  auto batch_sizes = std::vector<std::size_t>{
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, rows
  };
  // auto batch_sizes = std::vector<std::size_t>{1024};
  auto batch_timings = std::vector<std::vector<std::size_t>>(10);

  // Run benchmarks for each framework
  auto fil_output = std::vector<real_t>(2 * output.size());
  auto gpu_buffer = rmm::device_buffer{buffer.size() * sizeof(real_t), fil_model.get_stream()};
  cudaMemcpy(gpu_buffer.data(), buffer.data(), buffer.size() * sizeof(real_t), cudaMemcpyHostToDevice);
  auto gpu_out_buffer = rmm::device_buffer{fil_output.size() * sizeof(real_t), fil_model.get_stream()};

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      // run_fil(fil_model, cur_input, reinterpret_cast<real_t*>(gpu_out_buffer.data()));
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto fil_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  cudaMemcpy(fil_output.data(), gpu_out_buffer.data(), fil_output.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
  std::cout << "FIL: " << fil_output.data()[1] << ", " << fil_output.data()[half_index * 2 + 1] << ", " << fil_output.data()[output.size() * 2 - 1] << "\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      // run_fil(fil_model_sparse, cur_input, reinterpret_cast<real_t*>(gpu_out_buffer.data()));
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();

  start = std::chrono::high_resolution_clock::now();

  gpu_out_buffer = rmm::device_buffer{output.size() * sizeof(real_t), fil_model.get_stream()};

  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      // std::cout << gpu_buffer.size() / sizeof(real_t) << ", " << j * batch << "\n";
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<real_t*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        2
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  auto her_gpu_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  auto her_output = std::vector<real_t>(output.size());
  cudaMemcpy(her_output.data(), gpu_out_buffer.data(), her_output.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
  std::cout << "Herring: " << her_output.data()[0] << ", " << her_output.data()[half_index] << ", " << her_output.data()[output.size() - 1] << "\n";

  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<real_t*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        4
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }

  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<real_t*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        8
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[4].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }

  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<real_t*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        16
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[5].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }

  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(gpu_buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3(
        herring3_model_gpu,
        cur_input.data,
        reinterpret_cast<real_t*>(gpu_out_buffer.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        32
      );
    }
    kayak::cuda_check(cudaStreamSynchronize(fil_model.get_stream()));
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[6].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }

  std::fill(std::begin(output), std::end(output), 0);
  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      // std::cout << gpu_buffer.size() / sizeof(real_t) << ", " << j * batch << "\n";
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(buffer.data()) + j * batch * features, std::min(batch, rows - j * batch), features};
      run_herring3_cpu(
        herring3_model_cpu,
        cur_input.data,
        reinterpret_cast<real_t*>(output.data()),
        cur_input.rows,
        cur_input.cols,
        fil_model.get_stream(),
        64
      );
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[7].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Herring CPU: " << output.data()[0] << ", " << output.data()[half_index] << ", " << output.data()[output.size() - 1] << "\n";

  std::fill(std::begin(output), std::end(output), 0);
  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      // std::cout << gpu_buffer.size() / sizeof(real_t) << ", " << j * batch << "\n";
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(buffer.data()) + j * features * batch, std::min(batch, rows - j * batch), features};
      run_gtil(tl_model, cur_input, output);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[8].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "GTIL: " << output.data()[0] << ", " << output.data()[half_index] << ", " << output.data()[output.size() - 1] << "\n";

  std::fill(std::begin(output), std::end(output), 0);
  start = std::chrono::high_resolution_clock::now();
  for (auto i = std::size_t{}; i < batch_sizes.size(); ++i) {
    auto batch = batch_sizes[i];
    auto total_batches = rows / batch + (rows % batch != 0);
    // total_batches = 3;

    auto batch_start = std::chrono::high_resolution_clock::now();
    for (auto j = std::size_t{}; j < total_batches; ++j) {
      // std::cout << gpu_buffer.size() / sizeof(real_t) << ", " << j * batch << "\n";
      auto cur_input = matrix<real_t>{reinterpret_cast<real_t*>(buffer.data()) + j * features * batch, std::min(batch, rows - j * batch), features};
      // run_old_herring(old_herring_model, cur_input, output);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    batch_timings[9].push_back(std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count());
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Old Herring: " << output.data()[0] << ", " << output.data()[half_index] << ", " << output.data()[output.size() - 1] << "\n";

  std::cout << "FIL, Herring3" << "\n";
  std::cout << fil_elapsed << ", " << her_gpu_elapsed << "\n";
  std::cout << "Framework";
  for (auto size : batch_sizes) {
    std::cout << "," << size;
  }
  std::cout << "\n";
  std::cout << "FIL-Dense";
  for (auto res : batch_timings[0]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "FIL-Sparse";
  for (auto res : batch_timings[1]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H" << 2;
  for (auto res : batch_timings[2]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H" << 4;
  for (auto res : batch_timings[3]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H" << 8;
  for (auto res : batch_timings[4]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H" << 16;
  for (auto res : batch_timings[5]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H" << 32;
  for (auto res : batch_timings[6]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H3-CPU";
  for (auto res : batch_timings[7]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "GTIL";
  for (auto res : batch_timings[8]) {
    std::cout << "," << res;
  }
  std::cout << "\n";
  std::cout << "H1-CPU";
  for (auto res : batch_timings[9]) {
    std::cout << "," << res;
  }
  std::cout << "\n";

  return 0;
}
