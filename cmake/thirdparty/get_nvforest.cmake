#=============================================================================
# Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

function(find_and_configure_nvforest)

    set(oneValueArgs VERSION FORK PINNED_TAG USE_TREELITE_STATIC ENABLE_GPU)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(nvforest ${PKG_VERSION}
      GLOBAL_TARGETS      nvforest++
      BUILD_EXPORT_SET    rapids_triton-exports
      INSTALL_EXPORT_SET  rapids_triton-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/nvforest.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            GIT_SHALLOW    TRUE
            OPTIONS
              "BUILD_NVFOREST_TESTS OFF"
              "BUILD_SHARED_LIBS ON"
              "NVFOREST_ENABLE_GPU ${PKG_ENABLE_GPU}"
              "NVFOREST_USE_TREELITE_STATIC ${PKG_USE_TREELITE_STATIC}"
              "USE_CCACHE ON"
    )

    message(VERBOSE "RAPIDS_TRITON: Using nvForest located in ${nvforest_SOURCE_DIR}")

endfunction()

find_and_configure_nvforest(VERSION    26.06
                            FORK       rapidsai
                            PINNED_TAG a9216c6162daef1434fd4cfa554c7a963c6b2016
                            USE_TREELITE_STATIC ${TRITON_FIL_USE_TREELITE_STATIC}
                            ENABLE_GPU ${TRITON_ENABLE_GPU}
                            )
