# Copyright (c) 2021, NVIDIA CORPORATION.
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
#

cmake_minimum_required(VERSION 3.17)

project(tritonfilbackend LANGUAGES C CXX CUDA)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

find_path(RAFT_DIR "raft"
    HINTS
    "$ENV{RAFT_PATH}/cpp/include"
    "$ENV{CONDA_PREFIX}/include/cuml"
    "$ENV{CONDA_PREFIX}/include")

message(STATUS "RAFT: RAFT_DIR set to ${RAFT_DIR}")

find_package(Treelite 1.3.0 REQUIRED)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_fil.ldscript libtriton_fil.ldscript COPYONLY)

add_library(
  triton-fil-backend SHARED
  src/api.cu
  src/config.cc
  src/enum_conversions.cc
  src/exceptions.cc
  src/model_instance_state.cu
  src/model_state.cc
  src/triton_utils.cc
)

add_library(
  TritonFilBackend::triton-fil-backend ALIAS triton-fil-backend
)

target_include_directories(
  triton-fil-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RAFT_DIR}
)

target_compile_features(triton-fil-backend PRIVATE cxx_std_11)

target_compile_options(
  triton-fil-backend PRIVATE
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wno-type-limits
  "$<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>"
  "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-Werror>"
)

find_library(CUML_LIBRARY
  NAMES cuml++
  HINTS "$ENV{CONDA_PREFIX}/lib"
)

target_link_libraries(
  triton-fil-backend
  PUBLIC
    treelite::treelite
    ${CUML_LIBRARY}
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
)

set_target_properties(
  triton-fil-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_fil
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_fil.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_fil.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonFilBackend)

install(
  TARGETS
    triton-fil-backend
  EXPORT
    triton-fil-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fil
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fil
)

install(
  EXPORT
    triton-fil-backend-targets
  FILE
  TritonFilBackendTargets.cmake
  NAMESPACE
  TritonFilBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonFilBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonFilBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonFilBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-fil-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonFilBackendTargets.cmake
  NAMESPACE TritonFilBackend::
)

export(PACKAGE TritonFilBackend)
