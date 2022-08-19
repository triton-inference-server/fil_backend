#include <cstddef>
#include <herring3/constants.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <herring3/detail/predict.cuh>
#include <kayak/cuda_stream.hpp>
#include <kayak/tree_layout.hpp>
namespace herring {

template void predict<
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >
>(
  forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  > const&,
  postprocessor<float, float> const&,
  float*,
  float*,
  std::size_t,
  std::size_t,
  std::size_t,
  std::optional<std::size_t>,
  int device,
  kayak::cuda_stream stream
);

void initialize_gpu_options(int device) {
  using forest_t = forest<
    preferred_tree_layout, float, uint32_t, uint16_t, uint16_t, float
  >;
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer<false, false, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
}

}
