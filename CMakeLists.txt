# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
  GIT_REPOSITORY https://github.com/wphicks/common.git
  GIT_TAG bug-gcc9_compilation
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

find_package(Treelite 1.0.0 REQUIRED)
find_package(RMM REQUIRED MODULE)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_fil.ldscript libtriton_fil.ldscript COPYONLY)

add_library(
  triton-fil-backend SHARED
  src/api.cu
  src/c_wrappers.cpp
  src/config.cpp
  src/enum_conversions.cpp
  src/model_instance_state.cpp
  src/model_state.cpp
)

add_library(
  TritonFilBackend::triton-fil-backend ALIAS triton-fil-backend
)

target_include_directories(
  triton-fil-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RAFT_DIR}
    ${RMM_INCLUDE_DIRS}
)

target_compile_features(triton-fil-backend PRIVATE cxx_std_14)

target_compile_options(
  triton-fil-backend PRIVATE
  -Wall -Wextra -Wno-unused-parameter -Wno-type-limits
)

find_library(CUML_LIBRARY
  NAMES cuml++
  HINTS "$ENV{CONDA_PREFIX}/lib"
)

# target_link_directories(
#   triton-fil-backend
#   PUBLIC
#     "/root/miniconda3/envs/triton_dev/lib"
# )

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
