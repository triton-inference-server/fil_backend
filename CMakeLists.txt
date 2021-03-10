# specify version
cmake_minimum_required(VERSION 3.2)

# specify app
project(fil_backend LANGUAGES C CXX)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

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

#
# Shared library implementing the Triton Backend API
#
# TODO why do we do this? what is an ldscript?
configure_file(src/libtriton_fil.ldscript libtriton_fil.ldscript COPYONLY)

# Note to self: this creates the libtriton-fil-backend.so object
add_library(
        triton-fil-backend SHARED
        src/fil.cc
)

add_library(
        TritonFILBackend::triton-fil-backend ALIAS triton-fil-backend
)

target_include_directories(
        triton-fil-backend
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-fil-backend PRIVATE cxx_std_11)
target_compile_options(
        triton-fil-backend PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
)

# Note to self: remove -Werror from above - sets warnings as errors

##############################################################
# Note to self: Added these lines to include cuML
##############################################################
# include CUDA
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# create executable
add_executable(${PROJECT_NAME} src/fil.cc)

# find rapids library path
find_library(rapidsai-cuml-path cuml)

# include raft
include_directories("/home/whicks/proj_cuml_triton/cuml/cpp/build/raft/src/raft/cpp/include")

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        triton-backend-utils    # from repo-backend
        triton-core-serverstub  # from repo-core
        ${rapidsai-cuml-path}
)

target_link_libraries(
        triton-fil-backend
        PRIVATE
        triton-backend-utils    # from repo-backend
        triton-core-serverstub  # from repo-core
        ${rapidsai-cuml-path}
)
##############################################################


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
