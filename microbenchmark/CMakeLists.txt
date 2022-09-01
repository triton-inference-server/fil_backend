add_executable(forest_bench
  src/herring2/detail/infer/specializations/double_io_double_threshold_few_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/double_io_double_threshold_few_features.cu>
  src/herring2/detail/infer/specializations/double_io_double_threshold_many_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/double_io_double_threshold_many_features.cu>
  src/herring2/detail/infer/specializations/double_io_single_threshold_few_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/double_io_single_threshold_few_features.cu>
  src/herring2/detail/infer/specializations/double_io_single_threshold_many_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/double_io_single_threshold_many_features.cu>
  src/herring2/detail/infer/specializations/single_io_double_threshold_few_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/single_io_double_threshold_few_features.cu>
  src/herring2/detail/infer/specializations/single_io_double_threshold_many_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/single_io_double_threshold_many_features.cu>
  src/herring2/detail/infer/specializations/single_io_single_threshold_few_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/single_io_single_threshold_few_features.cu>
  src/herring2/detail/infer/specializations/single_io_single_threshold_many_features.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring2/detail/infer/specializations/single_io_single_threshold_many_features.cu>
  microbenchmark/src/forest_bench.cpp
)
add_executable(forest_bench3
  microbenchmark/src/forest_bench3.cpp
  src/herring3/specializations/device_initialization0.cu
  src/herring3/specializations/device_initialization1.cu
  src/herring3/specializations/device_initialization2.cu
  src/herring3/specializations/device_initialization3.cu
  src/herring3/specializations/infer0.cu
  src/herring3/specializations/infer0.cpp
  src/herring3/specializations/infer1.cu
  src/herring3/specializations/infer1.cpp
  src/herring3/specializations/infer2.cu
  src/herring3/specializations/infer2.cpp
  src/herring3/specializations/infer3.cu
  src/herring3/specializations/infer3.cpp
)

if(TRITON_ENABLE_GPU)
  set_target_properties(forest_bench
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(
    forest_bench PUBLIC ENABLE_GPU
  )
else()
  set_target_properties(forest_bench
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(forest_bench
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>
          $<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr>
)
target_include_directories(forest_bench
  PRIVATE "${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/microbenchmark/include"
)

if(TRITON_ENABLE_GPU)
  target_link_libraries(forest_bench
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      raft::raft
      cuml++
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost
  )
else()
  target_link_libraries(forest_bench
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost
  )
endif()
if(TRITON_ENABLE_GPU)
  set_target_properties(forest_bench3
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(
    forest_bench3 PUBLIC ENABLE_GPU
  )
else()
  set_target_properties(forest_bench3
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(forest_bench3
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>
          $<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr>
)
target_include_directories(forest_bench3
  PRIVATE "${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${CMAKE_CURRENT_SOURCE_DIR}/microbenchmark/include"
)

if(TRITON_ENABLE_GPU)
  target_link_libraries(forest_bench3
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      raft::raft
      cuml++
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost
  )
else()
  target_link_libraries(forest_bench3
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost
  )
endif()
