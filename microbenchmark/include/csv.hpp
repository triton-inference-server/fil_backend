#pragma once
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <utility>
#include <vector>
#include <exceptions.hpp>
#include <input_data.hpp>

template <typename T>
auto read_csv(std::filesystem::path const & csv_path) {
  auto storage = std::vector<T>{};
  auto rows = std::size_t{};
  auto cols = std::size_t{};
  {
    auto file_stream = std::ifstream{csv_path.c_str()};
    file_stream.seekg(0, std::ios::end);
    auto file_size = file_stream.tellg();
    auto raw_data = std::string(file_size, ' ');
    file_stream.seekg(0);
    file_stream.read(&raw_data[0], file_size);

    auto raw_stream = std::stringstream{raw_data};

    auto newline = std::string{"\n"};
    auto comma = std::string{","};

    auto row_string = std::string{};
    auto col_string = std::string{};
    std::getline(raw_stream, row_string);

    while(raw_stream.good()) {
      auto row_stream = std::stringstream{row_string};
      std::getline(row_stream, col_string, ',');
      cols = std::size_t{};
      while(row_stream.good()) {
        if constexpr (std::is_same_v<T, float>) {
          storage.push_back(std::stof(col_string));
        } else {
          storage.push_back(std::stod(col_string));
        }
        ++cols;
        std::getline(row_stream, col_string, ',');
      }
      ++rows;
      std::getline(raw_stream, row_string);
    }
  } // Limit scope of file data read into string

  auto temp_buffer = kayak::buffer<T>{
    storage.data(),
    storage.size()
  };

  return input_data<T>{
    kayak::buffer<T>{temp_buffer},
    rows,
    cols
  };
}
