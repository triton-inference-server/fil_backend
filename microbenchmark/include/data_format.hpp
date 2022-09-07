#pragma once
#include <string>
#include <opt_parser.hpp>

enum struct data_format_val { csv, bin_float, bin_double };

inline auto string_to_data_format(std::string const& type_string) {
  auto result = data_format_val{};
  if (type_string == "csv") {
    result = data_format_val::csv;
  } else if (type_string == "bin_float") {
    result = data_format_val::bin_float;
  } else if (type_string == "bin_double") {
    result = data_format_val::bin_double;
  } else {
    throw option_parsing_exception(type_string + " not recognized as a valid data format.");
  }
  return result;
}
