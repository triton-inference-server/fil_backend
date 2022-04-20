#pragma once
#include<herring2/device_type.hpp>
#include<type_traits>

namespace herring {
namespace detail {

template<device_type D, typename T>
struct owning_buffer {
};

}
}
