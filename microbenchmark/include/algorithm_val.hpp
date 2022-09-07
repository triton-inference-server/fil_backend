#pragma once
#include <string>
#include <opt_parser.hpp>

enum struct algorithm_val {
  herring_cpu,
  herring_gpu,
  fil_sparse,
  fil_dense,
  fil_dense_reorg,
  fil_sparse8,
  xgboost_cpu,
  xgboost_gpu
};

inline auto string_to_algorithm_val(std::string const& type_string) {
  auto result = algorithm_val{};
  if (type_string == "herring_cpu") {
    result = algorithm_val::herring_cpu;
  } else if (type_string == "herring_gpu") {
    result = algorithm_val::herring_gpu;
  } else if (type_string == "fil_sparse") {
    result = algorithm_val::fil_sparse;
  } else if (type_string == "fil_dense") {
    result = algorithm_val::fil_dense;
  } else if (type_string == "fil_dense_reorg") {
    result = algorithm_val::fil_dense_reorg;
  } else if (type_string == "fil_sparse8") {
    result = algorithm_val::fil_sparse8;
  } else if (type_string == "xgboost_cpu") {
    result = algorithm_val::xgboost_cpu;
  } else if (type_string == "xgboost_gpu") {
    result = algorithm_val::xgboost_gpu;
  } else {
    throw option_parsing_exception(type_string + " not recognized as a valid algorithm.");
  }
  return result;
}
