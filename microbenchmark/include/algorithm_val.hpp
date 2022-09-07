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
inline auto algorithm_val_to_str(algorithm_val const& alg) {
  auto result = std::string{};
  switch (alg) {
    case algorithm_val::herring_cpu:
      result = "herring_cpu";
      break;
    case algorithm_val::herring_gpu:
      result = "herring_gpu";
      break;
    case algorithm_val::fil_sparse:
      result = "fil_sparse";
      break;
    case algorithm_val::fil_dense:
      result = "fil_dense";
      break;
    case algorithm_val::fil_dense_reorg:
      result = "fil_dense_reorg";
      break;
    case algorithm_val::fil_sparse8:
      result = "fil_sparse8";
      break;
    case algorithm_val::xgboost_cpu:
      result = "xgboost_cpu";
      break;
    case algorithm_val::xgboost_gpu:
      result = "xgboost_gpu";
      break;
  }
  return result;
}
