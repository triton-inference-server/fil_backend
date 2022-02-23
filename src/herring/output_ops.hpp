#pragma once

namespace herring {

  /* Enum representing possible element-wise operations on output */
  enum class element_op {
    disable,
    signed_square,
    hinge,
    sigmoid,
    exponential,
    exponential_standard_ratio,
    logarithm_one_plus_exp
  };

  /* Enum representing possible row-wise operations on output */
  enum class row_op {
    disable,
    softmax,
    max_index
  };
}
