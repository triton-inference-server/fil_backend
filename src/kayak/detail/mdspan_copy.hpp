#pragma once
namespace kayak {
namespace detail {
  template <typename to_mdspan_t, typename from_mdspan_t>
  void mdspan_copy(
    to_mdspan_t& to_mdspan, from_mdspan_t const& from_mdspan
  ) {
    if constexpr(
      raft::is_device_accessible_v<to_mdspan_t>
      && raft::is_device_accessible_v<from_mdspan_t>
    ) {
      mdspan_copy::mdspan_copy<device_type::gpu>(to_mdspan, from_mdspan);
    } else {
      static_assert(
        raft::is_host_accessible_v<to_mdspan_t>
        && raft::is_host_accessible_v<from_mdspan_t>,
        "mdspan_copy must be applied to same-device data"
      );
      mdspan_copy::mdspan_copy<device_type::cpu>(to_mdspan, from_mdspan);
    }
    // TODO(wphicks)
  }
}
}
