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

function(find_and_configure_rapids_triton)

    set(oneValueArgs VERSION FORK PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(rapids_triton ${PKG_VERSION}
      GLOBAL_TARGETS      rapids_triton::rapids_triton
      BUILD_EXPORT_SET    rapids_triton_linear-exports
      INSTALL_EXPORT_SET  rapids_triton_linear-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/rapids-triton.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_EXAMPLE OFF"
    )

  message(VERBOSE "RAPIDS_TRITON_LINEAR: Using RAPIDS-Triton located in ${rapids_triton_SOURCE_DIR}")

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_rapids_triton(VERSION    21.10
                        FORK       rapidsai
                        PINNED_TAG branch-21.10
                        )
