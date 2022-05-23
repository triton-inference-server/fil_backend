#pragma once

namespace kayak {

/* Return the value that must be added to val to equal the next multiple of
 * alignment greater than or equal to val */
template <typename int_t>
auto padding_size(int_t val, int_t alignment) {
  auto result = val;
  if (alignment != 0) {
    result = alignment - (val % alignment);
  }
  return result;
}

/* Return the next multiple of alignment >= val */
template <typename int_t>
auto padded_size(int_t val, int_t alignment) {
  return val + padding_size(val, alignment);
}

/* Return the value that must be added to val to equal the next multiple of
 * alignment less than or equal to val */
template <typename int_t>
auto downpadding_size(int_t val, int_t alignment) {
  auto result = val;
  if (alignment != 0) {
    result = val % alignment;
  }
  return result;
}

/* Return the next multiple of alignment <= val */
template <typename int_t>
auto downpadded_size(int_t val, int_t alignment) {
  return val - downpadding_size(val, alignment);
}

}
