#pragma once

namespace kayak {
namespace detail {
  template <typename to_mdspan_t, typename from_mdspan_t>
  __global__ void mdspan_copy(
    to_mdspan_t& to_mdspan, from_mdspan_t const& from_mdspan
  ) {
    using from_layout = typename from_mdspan_t::layout_type;
    using to_layout = typename to_mdspan_t::layout_type;

    auto static constexpr const TILE_DIM = 32;
    auto static constexpr const TILE_SIZE = TILE_DIM * TILE_DIM;
    __shared__ typename from_mdspan_t::value_type tile[
      TILE_DIM * (TILE_DIM + 1)
    ];

    for (auto tile_start=0; tile_start < from_mdspan.size(); tile_start += TILE_SIZE) {
      // TODO: Handle a full tile in here
      auto i = blockIdx.x + tile_start;
      typename from_mdspan_t::size_type indices[2];
      if constexpr (std::is_same_v<from_layout, raft::layout_left>) {
        auto minor_dim = from_mdspan.extents()[0];
        indices[0] = i % minor_dim;
        indices[1] = i / minor_dim;
      } else {
        auto minor_dim = from_mdspan.extents()[1];
        indices[0] = i / minor_dim;
        indices[1] = i % minor_dim;
      }
      // TODO: Check if valid indices
      tile[compute_offset_index] = from_mdspan(indices[0], indices[1]);
      __syncthreads();
      // TODO: Copy back to output

    }
  }
}
}
