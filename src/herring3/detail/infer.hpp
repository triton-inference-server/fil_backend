#pragma once
#include <cstddef>
#include <optional>
#include <type_traits>
#include <herring3/detail/infer/cpu.hpp>
#ifdef ENABLE_GPU
#include <herring3/detail/infer/gpu.hpp>
#endif
#include <herring3/detail/postprocessor.hpp>
#include <herring3/exceptions.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
namespace herring {
namespace detail {

template<kayak::device_type D, typename forest_t>
void infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  std::size_t row_count,
  std::size_t col_count,
  std::size_t class_count,
  typename forest_t::io_type* vector_output=nullptr,
  std::optional<std::size_t> specified_chunk_size=std::nullopt,
  kayak::device_id<D> device=kayak::device_id<D>{},
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {
  if (vector_output == nullptr) {
    inference::infer<D, forest_t, std::nullptr_t> (
      forest,
      postproc,
      output,
      input,
      row_count,
      col_count,
      class_count,
      nullptr,
      specified_chunk_size,
      device,
      stream
    );
  } else {
    inference::infer<D, forest_t> (
      forest,
      postproc,
      output,
      input,
      row_count,
      col_count,
      class_count,
      vector_output,
      specified_chunk_size,
      device,
      stream
    );
  }
}

}
}
