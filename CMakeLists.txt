#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_FIL_DOCKER_BUILD OFF)
option(BUILD_SHARED_LIBS OFF)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

if(TRITON_FIL_DOCKER_BUILD)
  set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:23.08-py3" CACHE STRING "Build image for Dockerized builds")
  set(TRITON_BUILD_CONTAINER_VERSION "23.08" CACHE STRING "Triton version for Dockerized builds")

  add_custom_command(
    OUTPUT fil/libtriton_fil.so $<$<BOOL:${TRITON_ENABLE_GPU}>:fil/libcuml++.so>
    COMMAND DOCKER_BUILDKIT=1 docker build -t
      triton_fil_builder
      --build-arg BUILD_TYPE=${CMAKE_BUILD_TYPE}
      --build-arg TRITON_VERSION=${TRITON_BUILD_CONTAINER_VERSION}
      --build-arg BASE_IMAGE=${TRITON_BUILD_CONTAINER}
      --build-arg TRITON_ENABLE_GPU=${TRITON_ENABLE_GPU}
      --build-arg TRITON_ENABLE_STATS=${TRITON_ENABLE_GPU}
      --build-arg TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG}
      --build-arg TRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG}
      --build-arg TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG}
      -f ${CMAKE_CURRENT_LIST_DIR}/ops/Dockerfile
      ${CMAKE_CURRENT_LIST_DIR}
    COMMAND docker rm triton_fil_builder || echo 'error ignored..' || true
    COMMAND docker create --name triton_fil_builder triton_fil_builder
    COMMAND rm -rf fil
    COMMAND docker cp triton_fil_builder:/opt/tritonserver/backends/fil fil
    COMMAND docker rm triton_fil_builder
    COMMENT "Building FIL backend in Docker"
  )
  add_custom_target(
    fil_docker
    ALL
    DEPENDS fil/libtriton_fil.so $<$<BOOL:${TRITON_ENABLE_GPU}>:fil/libcuml++.so>
  )
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fil
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends
  )

else()
  cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

  ##############################################################################
  # - Target names -------------------------------------------------------------
  set(BACKEND_NAME "fil")
  set(BACKEND_TARGET "triton_${BACKEND_NAME}")


  ##############################################################################
  # - Prepare rapids-cmake -----------------------------------------------------
  file(DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.08/RAPIDS.cmake
      ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  ##############################################################################
  # - User Options  ------------------------------------------------------------

  option(BUILD_BACKEND_TESTS "Build RAPIDS_TRITON_BACKEND unit-tests" ON)
  option(CUDA_ENABLE_KERNEL_INFO "Enable kernel resource usage info" OFF)
  option(CUDA_ENABLE_LINE_INFO "Enable lineinfo in nvcc" OFF)
  option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
  option(DISABLE_DEPRECATION_WARNINGS "Disable depreaction warnings " ON)
  option(NVTX "Enable nvtx markers" OFF)
  set(BACKEND_FOLDER "/opt/tritonserver/backends" CACHE STRING "Triton backend folder path")
  set(RAPIDS_DEPENDENCIES_VERSION "23.08" CACHE STRING "RAPIDS projects dependencies version")
  option(TRITON_FIL_USE_TREELITE_STATIC "Link Treelite statically in libtriton_fil.so and cuml++.so" ON)


  message(VERBOSE "RAPIDS_TRITON_BACKEND: Enabling detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: Enabling kernelinfo in nvcc: ${CUDA_ENABLE_KERNEL_INFO}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: Enabling lineinfo in nvcc: ${CUDA_ENABLE_LINE_INFO}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: Enabling nvtx markers: ${NVTX}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: Build RAPIDS_TRITON_BACKEND unit-tests: ${BUILD_TESTS}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: RAPIDS projects dependencies version: ${RAPIDS_DEPENDENCIES_VERSION}")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: Linking Treelite statically: ${FIL_TRITON_USE_TREELITE_STATIC}")

  # Set RMM logging level
  set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
  set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
  message(VERBOSE "RAPIDS_TRITON_BACKEND: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

  ##############################################################################
  # - Project Initialization ---------------------------------------------------

  if(TRITON_ENABLE_GPU)
    rapids_cuda_init_architectures(RAPIDS_TRITON_BACKEND)
    project(RAPIDS_TRITON_BACKEND VERSION 22.10.00 LANGUAGES CXX CUDA)
  else()
    project(RAPIDS_TRITON_BACKEND VERSION 22.10.00 LANGUAGES CXX CUDA)
  endif()

  ##############################################################################
  # - build type ---------------------------------------------------------------

  # Set a default build type if none was specified
  rapids_cmake_build_type(Release)

  # this is needed for clang-tidy runs
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  ##############################################################################
  # - Conda environment detection ----------------------------------------------

  if(DETECT_CONDA_ENV)
    rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
        message(STATUS "RAPIDS_TRITON_BACKEND: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
    endif()
  endif()

  ##############################################################################
  # - compiler options ---------------------------------------------------------
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  if(TRITON_ENABLE_GPU)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

    # * find CUDAToolkit package
    # * determine GPU architectures
    # * enable the CMake CUDA language
    # * set other CUDA compilation flags
    rapids_find_package(CUDAToolkit REQUIRED
        BUILD_EXPORT_SET rapids_triton-exports
        INSTALL_EXPORT_SET rapids_triton-exports
        )
    include(cmake/modules/ConfigureCUDA.cmake)
  endif()

  ##############################################################################
  # - Requirements -------------------------------------------------------------

  # add third party dependencies using CPM
  rapids_cpm_init()
  find_package(OpenMP REQUIRED)

  # Version and Branch to use for RAPIDS projects dependencies
  # This project trails behind RAPIDS versions by one release
  set(RAPIDS_TRITON_MIN_VERSION_rapids_projects "${RAPIDS_DEPENDENCIES_VERSION}.00")
  set(RAPIDS_TRITON_BRANCH_VERSION_rapids_projects "${RAPIDS_DEPENDENCIES_VERSION}")

  include(cmake/thirdparty/get_cuml.cmake)
  include(cmake/thirdparty/get_rapids-triton.cmake)

  if(BUILD_TESTS)
    include(cmake/thirdparty/get_gtest.cmake)
  endif()

  ##############################################################################
  # - install targets-----------------------------------------------------------

  add_library(
    ${BACKEND_TARGET} SHARED
    src/api.cc
  )

  IF(TRITON_ENABLE_GPU)
    set_target_properties(${BACKEND_TARGET}
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_sources(${BACKEND_TARGET} PRIVATE src/detail/postprocess_gpu.cu)
  else()
    set_target_properties(${BACKEND_TARGET}
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
  endif()

  target_compile_options(${BACKEND_TARGET}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
  )

  target_include_directories(${BACKEND_TARGET}
    PRIVATE  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/include>"
             "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )

  if(TRITON_FIL_USE_TREELITE_STATIC)
    list(APPEND TREELITE_LIBS treelite::treelite_static treelite::treelite_runtime_static)
    list(APPEND TREELITE_LIBS_NO_PREFIX treelite_static treelite_runtime_static)
  else()
    list(APPEND TREELITE_LIBS treelite::treelite treelite::treelite_runtime)
    list(APPEND TREELITE_LIBS_NO_PREFIX treelite treelite_runtime)
  endif()

  target_link_libraries(${BACKEND_TARGET}
  PRIVATE
    cuml++
    ${TREELITE_LIBS}
    rapids_triton::rapids_triton
    triton-core-serverstub
    triton-backend-utils
    "${TRITONSERVER_LIB}"
    $<TARGET_NAME_IF_EXISTS:conda_env>
    OpenMP::OpenMP_CXX
  )

  list(APPEND BACKEND_TARGET "cuml++")

  if(NOT TRITON_FIL_USE_TREELITE_STATIC)
    list(APPEND BACKEND_TARGET ${TREELITE_LIBS_NO_PREFIX})
  endif()

  install(
    TARGETS ${BACKEND_TARGET}
    LIBRARY DESTINATION ${BACKEND_FOLDER}/${BACKEND_NAME}
  )

  ##############################################################################
  # - build test executable ----------------------------------------------------

  # TODO (wphicks)
  # if(BUILD_TESTS)
  #   include(test/CMakeLists.txt)
  # endif()
endif() # TRITON_FIL_DOCKER_BUILD
