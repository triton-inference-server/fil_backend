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
      BUILD_EXPORT_SET    rapids_triton-exports
      INSTALL_EXPORT_SET  rapids_triton-exports
        CPM_ARGS
            GIT_REPOSITORY ${PKG_FORK}
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_EXAMPLE OFF"
    )
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
set (RAPIDS_FORK https://github.com/rapidsai/rapids-triton.git)
set (REPO_TAG branch-${RAPIDS_DEPENDENCIES_VERSION})
message(STATUS "Setting repo tag to ${REPO_TAG} for rapids fork ${RAPIDS_FORK}")
# if Triton tag and organization is specified, change the fork and the repo
if (NOT RAPIDS_TRITON_REPO_PATH STREQUAL RAPIDS_FORK)
  set (RAPIDS_FORK ${RAPIDS_TRITON_REPO_PATH})
  message(STATUS "Setting repo fork to ${RAPIDS_FORK}")
endif()
if (NOT RAPIDS_TRITON_REPO_TAG STREQUAL "main")
  set (REPO_TAG ${RAPIDS_TRITON_REPO_TAG})
  message(STATUS "Setting repo tag to ${REPO_TAG}")
endif()

find_and_configure_rapids_triton(VERSION    ${RAPIDS_DEPENDENCIES_VERSION}
                                 FORK       ${RAPIDS_FORK}
                                 PINNED_TAG ${REPO_TAG}
                                 )
