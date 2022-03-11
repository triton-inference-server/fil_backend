add_executable(forest_bench
  microbenchmark/forest_bench.cpp
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
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
)
target_include_directories(forest_bench
  PRIVATE "${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
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
      xgboost::xgboost
  )
else()
  target_link_libraries(forest_bench
    PRIVATE
      treelite::treelite_static
      treelite::treelite_runtime_static
      $<TARGET_NAME_IF_EXISTS:conda_env>
      OpenMP::OpenMP_CXX
      xgboost::xgboost
  )
endif()
