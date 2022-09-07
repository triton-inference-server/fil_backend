#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <algorithm_val.hpp>
#include <binary_data.hpp>
#include <csv.hpp>
#include <data_format.hpp>
#include <herring3/treelite_importer.hpp>
#include <input_data.hpp>
#include <kayak/tree_layout.hpp>
#include <opt_parser.hpp>
#include <precision_val.hpp>
#include <rapids_triton/exceptions.hpp>
#include <results.hpp>
#include <run_fil.hpp>
#include <run_herring.hpp>
#include <serialization.h>
#include <tl_utils.h>

auto print_usage(std::string const& executable_path) {
  std::cerr \
    << "usage: " \
    << executable_path \
    << " [model_path] [data_path] [-f model_format] [-d data_format] [-p precision] [-b [b1...]] [-a [a1...]] [-r rows] [-c cols]\n"
    << "    -f model_format: one of 'xgboost_json' (default), 'xgboost', 'lightgbm', 'treelite_checkpoint'\n"
    << "    -d data_format: one of 'csv' (default), 'bin_float', 'bin_double'\n"
    << "    -p precision: one of 'float' (default) or 'double'\n"
    << "    -b [b1...]: batch sizes to profile\n"
    << "    -a [a1...]: algorithms to profile ('herring_cpu', 'herring_gpu', 'fil_sparse', 'fil_dense', 'fil_dense_reorg', 'fil_sparse8')\n"
    << "    -r rows: for binary data, the number of rows\n"
    << "    -c cols: for binary data, the number of columns\n";
}

int main(int argc, char** argv) {
  auto model_path = std::filesystem::path{};
  auto data_path = std::filesystem::path{};
  auto model_format = triton::backend::fil::SerializationFormat{};
  auto data_format = data_format_val{};
  auto input = std::variant<input_data<float>, input_data<double>>{};
  auto precision = precision_val{};
  auto batch_sizes = std::vector<std::size_t>{};
  auto algorithms = std::vector<algorithm_val>{algorithm_val::herring_gpu};
  auto rows = std::size_t{};
  auto cols = std::size_t{};
  // Parse the filepath to the model file
  try {
    model_path = std::filesystem::path{get_positional_value(argc, argv, 0)};
    if (!std::filesystem::exists(model_path)) {
      std::cerr << "ERROR: File " << model_path.c_str() << " does not exist.\n";
      print_usage(argv[0]);
      return 1;
    }

    // Parse the filepath to the file containing the test data
    data_path = std::filesystem::path{get_positional_value(argc, argv, 1)};
    std::cout << "DATA PATH: " << data_path << "\n";
    if (!std::filesystem::exists(data_path)) {
      std::cerr << "ERROR: File " << data_path.c_str() << " does not exist.\n";
      print_usage(argv[0]);
      return 1;
    }

    // Parse the model format
    try {
      auto model_format_opt = get_optional_value(argc, argv, "-f", 1);
      if (model_format_opt.has_value()) {
        model_format = triton::backend::fil::string_to_serialization((*model_format_opt)[0]);
      } else {
        model_format = triton::backend::fil::SerializationFormat::xgboost_json;
      }
    } catch(triton::backend::rapids::TritonException const& exc) {
      std::cerr << "ERROR: " << exc.what() << "\n";
      print_usage(argv[0]);
      return 1;
    }

    // Parse the data format
    auto data_format_opt = get_optional_value(argc, argv, "-d", 1);
    if (data_format_opt.has_value()) {
      data_format = string_to_data_format((*data_format_opt)[0]);
    } else {
      data_format = data_format_val::csv;
    }

    // Parse the number of rows and columns
    auto rows_opt = get_optional_value(argc, argv, "-r", 1);
    if (rows_opt.has_value()) {
      std::stringstream{(*rows_opt)[0]} >> rows;
    }
    auto cols_opt = get_optional_value(argc, argv, "-c", 1);
    if (cols_opt.has_value()) {
      std::stringstream{(*cols_opt)[0]} >> cols;
    }

    // Parse the desired model precision
    auto precision_opt = get_optional_value(argc, argv, "-p", 1);
    if (precision_opt.has_value()) {
      precision = string_to_precision((*precision_opt)[0]);
    } else {
      precision = precision_val::single_precision;
    }

    // Load the data
    // Note: We must actually execute the load here in order to determine the
    // number of rows if not supplied
    // TODO(wphicks)
    switch(data_format) {
      case data_format_val::csv:
        {
          if (precision == precision_val::single_precision) {
            input = read_csv<float>(data_path);
          } else {
            input = read_csv<double>(data_path);
          }
        }
        break;
      case data_format_val::bin_float:
        {
          if (rows == std::size_t{} || cols == std::size_t{}) {
            std::cerr << "ERROR: Rows and cols must be supplied for binary data\n";
          }
          if (precision == precision_val::single_precision) {
            input = read_binary_data<float, float>(data_path, rows, cols);
          } else {
            input = read_binary_data<float, double>(data_path, rows, cols);
          }
        }
        break;
      case data_format_val::bin_double:
        {
          if (rows == std::size_t{} || cols == std::size_t{}) {
            std::cerr << "ERROR: Rows and cols must be supplied for binary data\n";
          }
          if (precision == precision_val::single_precision) {
            input = read_binary_data<double, float>(data_path, rows, cols);
          } else {
            input = read_binary_data<double, double>(data_path, rows, cols);
          }
        }
        break;
    }

    // Parse the desired batch sizes
    auto batches_opt = get_optional_value(argc, argv, "-b");
    if (batches_opt.has_value()) {
      std::transform(
        std::begin(*batches_opt),
        std::end(*batches_opt),
        std::back_inserter(batch_sizes), 
        [](auto&& str_val) {
          auto stream = std::stringstream{str_val};
          auto result = std::size_t{};
          stream >> result;
          return result;
        }
      );
    } else {
      batch_sizes.push_back(std::visit([](auto&& concrete) { return concrete.rows; }, input));
    }

    // Parse the desired algorithms
    auto algorithms_opt = get_optional_value(argc, argv, "-a");
    if (algorithms_opt.has_value()) {
      algorithms.clear();
      std::transform(
        std::begin(*algorithms_opt),
        std::end(*algorithms_opt),
        std::back_inserter(algorithms), 
        string_to_algorithm_val
      );
    }
  } catch (option_parsing_exception const& exc) {
    std::cerr << "ERROR: " << exc.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }
  auto tl_model = load_tl_base_model(model_path, model_format);
  auto results = std::vector<benchmark_results>{};
  for (auto algo : algorithms) {
    auto algo_results = std::vector<benchmark_results>{};
    switch(algo) {
      case algorithm_val::herring_gpu:
        {
          auto model = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
            *tl_model,
            128u,
            precision == precision_val::double_precision,
            kayak::device_type::gpu,
            0,
            kayak::cuda_stream{}
          );
          algo_results = std::visit([&model, &batch_sizes](auto&& concrete_input) {
            return run_herring<kayak::device_type::gpu>(
              model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      case algorithm_val::herring_cpu:
        {
          auto model = herring::treelite_importer<kayak::tree_layout::depth_first>{}.import(
            *tl_model,
            64u,
            precision == precision_val::double_precision,
            kayak::device_type::cpu,
            0,
            kayak::cuda_stream{}
          );
          algo_results = std::visit([&model, &batch_sizes](auto&& concrete_input) {
            return run_herring<kayak::device_type::cpu>(
              model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      case algorithm_val::fil_sparse:
        {
          algo_results = std::visit([&tl_model, &batch_sizes](auto&& concrete_input) {
            return run_fil<algorithm_val::fil_sparse>(
              tl_model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      case algorithm_val::fil_dense:
        {
          algo_results = std::visit([&tl_model, &batch_sizes](auto&& concrete_input) {
            return run_fil<algorithm_val::fil_dense>(
              tl_model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      case algorithm_val::fil_dense_reorg:
        {
          algo_results = std::visit([&tl_model, &batch_sizes](auto&& concrete_input) {
            return run_fil<algorithm_val::fil_dense_reorg>(
              tl_model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      case algorithm_val::fil_sparse8:
        {
          algo_results = std::visit([&tl_model, &batch_sizes](auto&& concrete_input) {
            return run_fil<algorithm_val::fil_sparse8>(
              tl_model,
              concrete_input,
              batch_sizes
            );
          }, input);
        }
        break;
      default:
        std::cerr << "ERROR: algorithm " << algorithm_val_to_str(algo) << " not yet supported.\n";
        break;
    }
    std::move(std::begin(algo_results), std::end(algo_results), std::back_inserter(results));
  }
  std::cout << "Framework";
  std::for_each(std::begin(batch_sizes), std::end(batch_sizes), [](auto&& batch) {
    std::cout << "," << batch;
  });
  std::cout << "\n";
  std::for_each(std::begin(results), std::end(results), [](auto&& algo_result) {
    std::cout << algo_result.label;
    std::for_each(
      std::begin(algo_result.elapsed_times),
      std::end(algo_result.elapsed_times),
      [](auto&& duration) {
        std::cout << "," << duration;
      }
    );
    std::cout << "\n";
  });

  return 0;
}
