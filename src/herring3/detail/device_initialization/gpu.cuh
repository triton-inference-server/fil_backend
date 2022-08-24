#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <herring3/constants.hpp>
#include <herring3/detail/gpu_introspection.hpp>
#include <herring3/detail/infer_kernel/gpu.cuh>
#include <herring3/detail/forest.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_setter.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
namespace herring {
namespace detail {
namespace device_initialization {

template<typename forest_t, kayak::device_type D>
std::enable_if_t<kayak::GPU_ENABLED && D==kayak::device_type::gpu, void> initialize_device(kayak::device_id<D> device) {
  auto device_context = kayak::device_setter(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // TODO: Include all infer variants
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
}
extern template void initialize_device<
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >,
  kayak::device_type::gpu
>(kayak::device_id<kayak::device_type::gpu> device);

}
}
}

