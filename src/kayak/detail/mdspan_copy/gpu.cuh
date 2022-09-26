#pragma once

namespace kayak {
namespace detail {
  auto static constexpr const MDSPAN_COPY_TILE_DIM = 32;
  auto static constexpr const MDSPAN_COPY_TILE_SIZE = (
    MDSPAN_COPY_TILE_DIM * MDSPAN_COPY_TILE_DIM
  );

  template <typename to_mdspan_t, typename from_mdspan_t>
  __global__ void mdspan_copy(
    to_mdspan_t& to_mdspan, from_mdspan_t const& from_mdspan
  ) {
    // NOTE: This kernel should always be launched with MDSPAN_COPY_TILE_SIZE
    // threads per block
    using from_layout = typename from_mdspan_t::layout_type;
    using to_layout = typename to_mdspan_t::layout_type;

    __shared__ typename from_mdspan_t::value_type tile[
      MDSPAN_COPY_TILE_DIM * (MDSPAN_COPY_TILE_DIM + 1)
    ];

    for (
        auto tile_start=blockIdx.x;
        tile_start < from_mdspan.size();
        tile_start += MDSPAN_COPY_TILE_SIZE * blockDim.x
    ) {
      auto i = threadIdx.x + tile_start;

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
      auto real_thread = (
        indices[0] < from_mdspan.extents()[0]
        && indices[1] < from_mdspan.extents()[1]
      );
      // TODO(wphicks): More than one element per thread to amortize indexing
      // cost
      if (real_thread) {
        tile[threadIdx.x] = from_mdspan(indices[0], indices[1]);
      }
      __syncthreads();
      if (real_thread) {
        to_mdspan(indices[0], indices[1]) = tile[threadIdx.x];
      }
      __syncthreads();
    }
  }
}
}
