#pragma once
namespace kayak {
namespace detail {
namespace mdspan_copy {
  template <typename to_mdspan_t, typename from_mdspan_t>
  std::enable_if_t<
    ENABLE_GPU
    && raft::is_device_accessible_mdspan_v<to_mdspan_t>
    && raft::is_device_accessible_mdspan_v<from_mdspan_t>
  > mdspan_copy(to_mdspan_t& to_mdspan, from_mdspan_t const& from_mdspan);
}
}
}
