#pragma once
#include <stddef.h>
#include <limits>
#include <type_traits>
#include <kayak/gpu_support.hpp>

namespace herring {
  /* Enum representing possible row-wise operations on output */
  enum struct row_op : unsigned char {
    disable=0b00100000,
    softmax=0b01000000,
    max_index=0b10000000
  };

  /* Enum representing possible element-wise operations on output */
  enum struct element_op : unsigned char {
    disable=0b00000000,
    signed_square=0b00000001,
    hinge=0b00000010,
    sigmoid=0b00000100,
    exponential=0b00001000,
    logarithm_one_plus_exp=0b00010000
  };

  HOST DEVICE inline auto constexpr ops_to_val(row_op row_wise, element_op elem_wise) {
    return (
      static_cast<std::underlying_type_t<row_op>>(row_wise) |
      static_cast<std::underlying_type_t<element_op>>(elem_wise)
    );
  }

  template<
    row_op row_wise_v,
    element_op elem_wise_v,
    typename leaf_output_t,
    typename io_t
  >
  HOST DEVICE void postprocess(
    leaf_output_t* val,
    size_t class_count,
    io_t* out,
    io_t average_factor=io_t{1},
    io_t bias=io_t{0},
    io_t constant=io_t{1}
  ) {
    auto max_index = size_t{};
    auto max_value = std::numeric_limits<io_t>::lowest();
    for (auto i=size_t{}; i < class_count; ++i) {
      val[i] = val[i] / average_factor + bias;
      if constexpr (elem_wise_v == element_op::signed_square) {
        val[i] = copysign(val[i] * val[i], val[i]);
      } else if constexpr (elem_wise_v == element_op::hinge) {
        val[i] = leaf_output_t(val[i] > leaf_output_t{});
      } else if constexpr (elem_wise_v == element_op::sigmoid) {
        val[i] = leaf_output_t{1} / (leaf_output_t{1} + exp(-constant * val[i]));
      } else if constexpr (elem_wise_v == element_op::exponential) {
        val[i] = exp(val[i] / constant);
      } else if constexpr (elem_wise_v == element_op::logarithm_one_plus_exp) {
        val[i] = log1p(exp(val[i] / constant));
      }
      if constexpr (row_wise_v == row_op::softmax || row_wise_v == row_op::max_index) {
        auto is_new_max = val[i] > max_value;
        max_index = is_new_max * i + (!is_new_max) * max_index;
        max_value = is_new_max * val[i] + (!is_new_max) * max_value;
      }
    }

    if constexpr (row_wise_v == row_op::max_index) {
      *out = max_index;
    } else {
      for (auto i=size_t{}; i < class_count; ++i) {
        if constexpr (row_wise_v == row_op::softmax) {
          out[i] = exp(val[i] - max_value);
        } else {
          out[i] = val[i];
        }
      }
    }
  }

  template <typename leaf_output_t, typename io_t>
  struct postprocessor {
    HOST DEVICE postprocessor(
      row_op row_wise=row_op::disable,
      element_op elem_wise=element_op::disable,
      io_t average_factor=io_t{1},
      io_t bias=io_t{0},
      io_t constant=io_t{1}
    ) :
      average_factor_{average_factor},
      bias_{bias},
      constant_{constant},
      row_wise_{row_wise},
      elem_wise_{elem_wise} {
    }

    HOST DEVICE void operator()(leaf_output_t* val, size_t class_count, io_t* out) const {
      switch(ops_to_val(row_wise_, elem_wise_)) {
        case ops_to_val(row_op::disable, element_op::signed_square):
          postprocess<row_op::disable, element_op::signed_square>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::disable, element_op::hinge):
          postprocess<row_op::disable, element_op::hinge>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::disable, element_op::sigmoid):
          postprocess<row_op::disable, element_op::sigmoid>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::disable, element_op::exponential):
          postprocess<row_op::disable, element_op::exponential>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::disable, element_op::logarithm_one_plus_exp):
          postprocess<row_op::disable, element_op::logarithm_one_plus_exp>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::disable):
          postprocess<row_op::softmax, element_op::disable>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::signed_square):
          postprocess<row_op::softmax, element_op::signed_square>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::hinge):
          postprocess<row_op::softmax, element_op::hinge>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::sigmoid):
          postprocess<row_op::softmax, element_op::sigmoid>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::exponential):
          postprocess<row_op::softmax, element_op::exponential>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::softmax, element_op::logarithm_one_plus_exp):
          postprocess<row_op::softmax, element_op::logarithm_one_plus_exp>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::disable):
          postprocess<row_op::max_index, element_op::disable>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::signed_square):
          postprocess<row_op::max_index, element_op::signed_square>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::hinge):
          postprocess<row_op::max_index, element_op::hinge>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::sigmoid):
          postprocess<row_op::max_index, element_op::sigmoid>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::exponential):
          postprocess<row_op::max_index, element_op::exponential>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        case ops_to_val(row_op::max_index, element_op::logarithm_one_plus_exp):
          postprocess<row_op::max_index, element_op::logarithm_one_plus_exp>(
            val, class_count, out, average_factor_, bias_, constant_
          );
          break;
        default:
          postprocess<row_op::disable, element_op::disable>(
            val, class_count, out, average_factor_, bias_, constant_
          );
      }
    }
   private:
    io_t average_factor_;
    io_t bias_;
    io_t constant_;
    row_op row_wise_;
    element_op elem_wise_;
  };
}
