#pragma once
#include <herring2/device_type.hpp>
#include <herring2/device_id.hpp>

namespace herring {
namespace detail {

/** Struct for setting current device within a code block */
template <device_type D>
struct device_setter {
  device_setter(device_id<D> device) {}
};

}
}
