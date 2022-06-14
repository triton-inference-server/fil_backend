#pragma once
#include <nvtx3/nvtx3.hpp>
#include <cmath>
#include <herring/output_ops.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/flat_array.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/ndarray.hpp>

namespace herring {
namespace detail {
namespace inference {

using kayak::raw_index_t;

template <typename io_t>
HOST DEVICE auto postprocess(
  kayak::ndarray<io_t, 2u, 0u, 1u>& workspace,
  kayak::raw_index_t row_index,
  element_op element_postproc,
  row_op row_postproc,
  io_t average_factor,
  io_t bias,
  io_t postproc_constant
) {
  auto max_index = raw_index_t{};
  auto max_value = workspace.at(row_index, max_index, 0u);

  for (auto class_index = raw_index_t{}; class_index < workspace.dims()[1]; ++class_index) {
    for (auto grove_index = raw_index_t{1}; grove_index < workspace.dims()[2]; ++grove_index) {
      workspace.at(row_index, class_index, 0u) += workspace.at(row_index, class_index, grove_index);
    }
    workspace.at(row_index, class_index, 0u) = workspace.at(row_index, class_index, 0u) / average_factor + bias;
    auto elem = workspace.at(row_index, class_index, 0u);
    // Apply element-wise postprocessing
    switch(element_postproc) {
      case element_op::signed_square:
        workspace.at(row_index, class_index, 0u) = std::copysign(elem * elem, elem);
        break;
      case element_op::hinge:
        workspace.at(row_index, class_index, 0u) = elem > io_t{} ? io_t{1} : io_t{0};
        break;
      case element_op::sigmoid:
        workspace.at(row_index, class_index, 0u) = io_t{1} / (io_t{1} + std::exp(-postproc_constant * elem));
        break;
      case element_op::exponential:
        workspace.at(row_index, class_index, 0u) = std::exp(elem);
        break;
      case element_op::exponential_standard_ratio:
        workspace.at(row_index, class_index, 0u) = std::exp(-elem / postproc_constant);
        break;
      case element_op::logarithm_one_plus_exp:
        workspace.at(row_index, class_index, 0u) = std::log1p(std::exp(elem));
        break;
      default:
        break;
    }
    if (workspace.at(row_index, class_index, 0u) > max_value) {
      max_value = workspace.at(row_index, class_index, 0u);
      max_index = class_index;
    }
  }

  auto result = kayak::flat_array<kayak::array_encoding::dense, io_t>{
    &(workspace.at(row_index, 0u, 0u)), workspace.dims()[1]
  };
  switch(row_postproc) {
    case row_op::max_index:
      workspace.at(row_index, 0u, 0u) = max_index;
      result = kayak::flat_array<kayak::array_encoding::dense, io_t>{
        &(workspace.at(row_index, 0u, 0u)), 1u
      };
      break;
    case row_op::softmax:
      {
        auto normalization = io_t{};
        for (auto class_index = raw_index_t{}; class_index < workspace.dims()[1]; ++class_index) {
          workspace.at(row_index, class_index, 0u) = std::exp(
            workspace.at(row_index, class_index, 0u) - max_value
          );
          normalization += workspace.at(row_index, class_index, 0u);
        }
        for (auto class_index = raw_index_t{}; class_index < workspace.dims()[1]; ++class_index) {
          workspace.at(row_index, class_index, 0u) /= normalization;
        }
      }
      break;
    default:
      break;
  }
  return result;
}

}
}
}
