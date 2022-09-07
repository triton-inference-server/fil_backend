#pragma once
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <kayak/buffer.hpp>
#include <input_data.hpp>

template <typename file_data_type, typename dest_data_type=file_data_type>
auto read_binary_data(std::filesystem::path const& path, std::size_t rows, std::size_t cols) {
  auto in_data = kayak::buffer<file_data_type>{rows * cols};
  auto* read_buffer = reinterpret_cast<char*>(in_data.data());
  auto in_stream = std::ifstream(path.c_str(), std::ifstream::binary);
  in_stream.read(read_buffer, in_data.size() * sizeof(file_data_type));
  if constexpr (!std::is_same_v<file_data_type, dest_data_type>) {
    auto out_data = kayak::buffer<dest_data_type>{rows * cols};
    std::copy(in_data.data(), in_data.data() + in_data.size(), out_data.data());
    return input_data<dest_data_type>{
      out_data,
      rows,
      cols
    };
  } else {
    return input_data<dest_data_type>{
      in_data,
      rows,
      cols
    };
  }
}
