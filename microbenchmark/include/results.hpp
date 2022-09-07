#pragma once
#include <string>
#include <vector>

struct benchmark_results {
  benchmark_results(std::string const& label_, std::vector<std::size_t> const& times_) : label{label_}, elapsed_times{times_} {}
  std::string label;
  std::vector<std::size_t> elapsed_times;
};
