#pragma once
#include <algorithm>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>

struct option_parsing_exception : std::exception {
  option_parsing_exception () : msg_{"Could not parse options"}
  {
  }
  option_parsing_exception (std::string msg) : msg_{msg}
  {
  }
  option_parsing_exception (char const* msg) : msg_{msg}
  {
  }
  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

inline auto get_optional_value(int argc, char** argv, std::string const& flag, std::optional<std::size_t> num_values=std::nullopt) {
  auto result = std::optional<std::vector<std::string>>{};
  auto arg_end = argv + argc;
  auto flag_pos = std::find(argv, arg_end, flag);
  if (flag_pos != arg_end) {
    result = std::vector<std::string>{};
    auto opt_pos = flag_pos;
    while (++opt_pos != arg_end && (*opt_pos)[0] != '-') {
      result->push_back(*opt_pos);
    }
    if (result->size() == 0) {
      throw option_parsing_exception{
        "Flag '" + flag + "' should be followed by its value(s)"
      };
    }
    if (result->size() != num_values.value_or(result->size())) {
      throw option_parsing_exception{
        "Flag '" \
          + flag \
          + "' should be followed by " \
          + std::to_string(num_values.value()) \
          + " value(s)"
      };
    }
  }
  return result;
}

inline auto get_positional_value(int argc, char** argv, std::size_t index) {
  if (index + 1 >= argc) {
    throw option_parsing_exception{
      "Expected at least " + std::to_string(index + 1) + " initial arguments"
    };
  }
  return std::string{argv[index + 1]};
}
