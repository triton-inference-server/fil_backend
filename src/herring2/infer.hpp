#pragma once

#include <herring2/decision_forest.hpp>
#include <kayak/data_array.hpp>
#include <kayak/detail/index_type.hpp>
#include <kayak/gpu_support.hpp>
#include <kayak/stream_pool.hpp>
#include <utility>

namespace herring {

template <
  kayak::device_type D,
  typename output_t,
  typename input_t,
  kayak::data_layout out_layout=kayak::data_layout::dense_row_major,
  kayak::data_layout in_layout=kayak::data_layout::dense_row_major
>
void predict(
  kayak::stream_pool<D> const& pool,
  forest_model_variant<output_t> const& model,
  kayak::data_array<out_layout, output_t>& out,
  kayak::data_array<in_layout, input_t> const& in,
  bool predict_proba = false
) {
  detail::predict<D>(pool, model.obj(), out, in, predict_proba);
}

/** Perform forest inference on given inputs, storing result in out
 *
 * This prediction function assumes that inputs and outputs are stored on the
 * indicated device. If D is device_type::gpu, it further assumes that inputs
 * and outputs are allocated on the current cuda device.
 */
template <
  kayak::device_type D,
  typename output_t,
  typename input_t,
  kayak::data_layout out_layout=kayak::data_layout::dense_row_major,
  kayak::data_layout in_layout=kayak::data_layout::dense_row_major
>
void predict(
  kayak::stream_pool<D> const& pool,
  forest_model_variant<output_t> const& model,
  output_t* out,
  input_t const* in,
  kayak::detail::index_type<kayak::DEBUG_ENABLED> row_count,
  bool predict_proba = false
) {
  auto output = kayak::data_array<out_layout, output_t>(
    out, row_count, model.outputs_per_sample()
  );
  auto input = kayak::data_array<in_layout, output_t>(
    in, row_count, model.num_features()
  );
  predict(pool, model, output, input, predict_proba);
}

template <
  kayak::device_type D,
  typename output_t,
  typename input_t,
  kayak::data_layout out_layout=kayak::data_layout::dense_row_major,
  kayak::data_layout in_layout=kayak::data_layout::dense_row_major
>
auto predict(
  forest_model_variant<output_t> const& model,
  output_t* out,
  input_t const* in,
  kayak::detail::index_type<kayak::DEBUG_ENABLED> row_count,
  bool predict_proba = false
) {
  auto output = kayak::data_array<out_layout, output_t>(
    out, row_count, model.outputs_per_sample()
  );
  auto input = kayak::data_array<in_layout, output_t>(
    in, row_count, model.num_features()
  );
  auto pool = kayak::stream_pool<D>{};
  predict(pool, model, output, input, predict_proba);
  return pool;
}

}
