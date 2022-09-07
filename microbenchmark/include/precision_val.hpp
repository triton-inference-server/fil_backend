#pragma once
#include <string>
#include <opt_parser.hpp>
enum struct precision_val { single_precision, double_precision };

template <precision_val P>
struct precision_val_to_type {
  using type = std::conditional<P==precision_val::single_precision, float, double>;
};
template <precision_val P>
using precision_val_t = typename precision_val_to_type<P>::type;

inline auto string_to_precision(std::string const& type_string) {
  auto result = precision_val{};
  if (type_string == "float" || type_string == "single") {
    result = precision_val::single_precision;
  } else if (type_string == "double") {
    result = precision_val::double_precision;
  } else {
    throw option_parsing_exception(type_string + " not recognized as a valid precision specifier.");
  }
  return result;
}
