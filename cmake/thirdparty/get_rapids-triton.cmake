#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

# TODO(hcho3): Revert PINNED_TAG when RAPIDS is upgraded to 25.12
find_and_configure_rapids_triton(VERSION    ${RAPIDS_DEPENDENCIES_VERSION}
                                 FORK       ${RAPIDS_TRITON_REPO_PATH}
                                 PINNED_TAG 661cbc8998cadecec7ccfbdea81ad4f1d6abf98f
                                 )
