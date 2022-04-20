#pragma once
#include <herring2/device_type.hpp>

namespace herring {
namespace detail {
template<device_type D>
struct device_id {
  using value_type = int;

  device_id(value_type device_index) {}
  auto value() const { return value_type{}; }
};
}
}
