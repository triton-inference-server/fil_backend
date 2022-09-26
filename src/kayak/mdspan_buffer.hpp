#pragma once
#include <kayak/buffer.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_type.hpp>
#include <raft/core/device_mdspan.h>
#include <raft/core/host_mdspan.h>
#include <raft/core/mdspan_types.h>

namespace kayak {

namespace detail {
  template <typename mdspan_t>
  auto get_mdspan_buffer_size(mdspan_t const& from_mdspan) {
    using layout_type = typename mdspan_t::layout_type;
    static_assert(
      std::is_same_v<layout_type, raft::layout_left>
      || std::is_same_v<layout_type, raft::layout_right>
      || std::is_same_v<layout_type, raft::layout_strided>,
      "Unable to determine buffer size required for given layout"
    );
    auto result = std::size_t{1};
    for (auto i = std::size_t{}; i < from_mdspan.rank()) {
      result *= from_mdspan.extents(i) * from_mdspan.stride(i);
    }
    return result;
  }
}

  template <typename mdspan_t>
  struct mdspan_buffer {

    using value_type = typename mdspan_t::value_type;
    using layout_type = typename mdspan_t::layout_type;

    template <typename from_mdspan_t>
    mdspan_buffer(
      from_mdspan_t const& from_mdspan,
      int device = 0,
      cuda_stream stream = cuda_stream{}
    ) : mdspan_buffer{[&from_mdspan, device, stream]() {
      auto constexpr to_device_type = (
        raft::is_device_accessible_mdspan_v<mdspan_t> ? device_type::gpu : device_type::cpu
      );
      auto constexpr from_device_type = []() constexpr {
        if constexpr (to_device_type == device_type::gpu && raft::is_device_accessible_mdspan_v<from_mdspan_t>) {
          return device_type::gpu;
        }
        if constexpr (to_device_type == device_type::cpu && raft::is_host_accessible_mdspan_v<from_mdspan_t>) {
          return device_type::cpu;
        }
        if constexpr (raft::is_device_accessible_mdspan_v<from_mdspan_t>) {
          return device_type::gpu;
        }
        return device_type::cpu;
      };

      auto result_buffer = buffer<typename mdspan_t::value_type>{};
      auto result_mdspan = mdspan_t{};

      auto constexpr requires_type_conversion = !std::is_same_v<typename from_mdspan_t::value_type, value_type>;
      auto constexpr requires_layout_conversion = !std::is_same_v<typename from_mdspan_t::layout_type, layout_type>;
      auto constexpr requires_access_conversion = (
        (raft::is_device_accessible_mdspan_v<mdspan_t> && !raft::is_device_accessible_mdspan_v<from_mdspan_t>)
        || (raft::is_host_accessible_mdspan_v<mdspan_t> && !raft::is_host_accessible_mdspan_v<from_mdspan_t>)
      );
      auto constexpr requires_conversion = (
        requires_type_conversion || requires_layout_conversion || requires_access_conversion
      );

      if constexpr (requires_conversion) {
        static_assert(
          !(raft::is_device_accessible_mdspan_v<mdspan_t> && raft::is_host_accessible_mdspan_v<mdspan_t>),
          "Managed memory not yet supported for kayak buffers"
        );
        result_mdspan = mdspan_t{
          static_cast<value_type*>(nullptr),
          from_mdspan.extents()
        };
        result_buffer = buffer<value_type>{
          get_mdspan_buffer_size(result_mdspan),
          to_device_type,
          device,
          stream
        };
        result_mdspan = mdspan_t{
          result_buffer.data(),
          from_mdspan.extents()
        };
        if constexpr (raft::is_device_accessible_mdspan_v<mdspan_t> && !raft::is_device_accessible_mdspan_v<from_mdspan_t>) {
          auto from_buffer = buffer{
            buffer{
              from_mdspan.data_handle(),
              get_mdspan_buffer_size(from_mdspan),
              from_device_type,
              device
            },
            to_device_type,
            device,
            stream
          };
          auto device_from_mdspan = raft::mdspan<
            typename from_mdspan_t::element_type,
            typename from_mdspan_t::extents_type,
            typename from_mdspan_t::layout_type,
            raft::device_accessor<typename from_mdspan_t::accessor_type>
          >{
            from_buffer.data(),
            from_mdspan.extents()
          }
          detail::mdspan_copy(
            result_mdspan,
            device_from_mdspan,
            stream
          );
        } else if constexpr (!raft::is_device_accessible_mdspan_v<mdspan_t> && raft::is_device_accessible_mdspan_v<from_mdspan_t>) {
          auto to_buffer = buffer{
            result_buffer,
            from_device_type,
            device,
            stream
          };
          auto device_to_mdspan = raft::mdspan<
            typename to_mdspan_t::element_type,
            typename to_mdspan_t::extents_type,
            typename to_mdspan_t::layout_type,
            raft::device_accessor<typename from_mdspan_t::accessor_type>
          >{
            to_buffer.data(),
            from_mdspan.extents()
          }
          detail::mdspan_copy(
            device_to_mdspan,
            from_mdspan,
            stream
          );
          copy(result_buffer, to_buffer, stream);
        } else {
          detail::mdspan_copy(
            result_mdspan,
            from_mdspan,
            stream
          );
        }
      } else {
        result_buffer = buffer<value_type>{
          from_mdspan.data_handle(),
          from_mdspan.size(),
          to_device_type,
          device
        };
        result_mdspan = from_mdspan;
      }
      return mdspan_buffer{result_buffer, result_mdspan};
    } ()} {}

    auto& get_mdspan() { return mdspan_; }
    auto const& get_mdspan() const { return mdspan_; }

   private:
    mdspan_buffer(
      buffer<typename mdspan_t::value_type>&& from_buffer,
      mdspan_t&& from_mdspan
    ) : buffer_{from_buffer}, mdspan_{from_mdspan} {}
    buffer<typename mdspan_t::value_type> buffer_;
    mdspan_t mdspan_;
  };

}
