add_executable(forest_bench
  microbenchmark/src/forest_bench.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring3/specializations/infer0.cu>
  src/herring3/specializations/infer0.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring3/specializations/infer1.cu>
  src/herring3/specializations/infer1.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring3/specializations/infer2.cu>
  src/herring3/specializations/infer2.cpp
  $<$<BOOL:${TRITON_ENABLE_GPU}>:src/herring3/specializations/infer3.cu>
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
      rapids_triton::rapids_triton
      cuml++
      OpenMP::OpenMP_CXX
      # xgboost
  )
else()
  target_link_libraries(forest_bench
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      rapids_triton::rapids_triton
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost
  )
endif()
