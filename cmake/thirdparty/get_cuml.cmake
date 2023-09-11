#=============================================================================
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
#=============================================================================

function(find_and_configure_cuml)

    set(oneValueArgs VERSION FORK PINNED_TAG USE_TREELITE_STATIC TRITON_ENABLE_GPU)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    set(CUML_ALGORITHMS "FIL" CACHE STRING "List of algorithms to build in cuml")
    list(APPEND CUML_ALGORITHMS "TREESHAP")

    rapids_cpm_find(cuml ${PKG_VERSION}
      GLOBAL_TARGETS      cuml++
      BUILD_EXPORT_SET    rapids_triton-exports
      INSTALL_EXPORT_SET  rapids_triton-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/cuml.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_CUML_C_LIBRARY OFF"
              "BUILD_CUML_CPP_LIBRARY ON"
              "BUILD_CUML_TESTS OFF"
              "BUILD_PRIMS_TESTS OFF"
              "BUILD_CUML_MG_TESTS OFF"
              "BUILD_CUML_EXAMPLES OFF"
              "BUILD_CUML_BENCH OFF"
              "BUILD_CUML_PRIMS_BENCH OFF"
              "BUILD_CUML_STD_COMMS OFF"
              "BUILD_SHARED_LIBS ON"
              "CUML_USE_TREELITE_STATIC ${PKG_USE_TREELITE_STATIC}"
              "CUML_ENABLE_GPU ${PKG_TRITON_ENABLE_GPU}"
              "USE_CCACHE ON"
              "RAFT_COMPILE_LIBRARIES OFF"
              "RAFT_ENABLE_NN_DEPENDENCIES OFF"
    )

    message(VERBOSE "RAPIDS_TRITON: Using CUML located in ${cuml_SOURCE_DIR}")

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_cuml(VERSION    ${RAPIDS_TRITON_MIN_VERSION_rapids_projects}
                        FORK       hcho3
                        PINNED_TAG fix_cpu_fil
                        USE_TREELITE_STATIC ${TRITON_FIL_USE_TREELITE_STATIC}
                        TRITON_ENABLE_GPU ${TRITON_ENABLE_GPU})
