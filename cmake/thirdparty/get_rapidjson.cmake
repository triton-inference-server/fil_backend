
   
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

# TODO(wphicks): Pass in version
function(find_and_configure_rapidjson VERSION)

    rapids_cpm_find(rapidjson ${VERSION}
        GLOBAL_TARGETS      rapidjson::rapidjson
        BUILD_EXPORT_SET    rapids_triton-exports
        INSTALL_EXPORT_SET  rapids_triton-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/Tencent/rapidjson
            GIT_TAG "v${VERSION}"
            GIT_SHALLOW ON
            OPTIONS
              "RAPIDJSON_BUILD_DOC OFF"
              "RAPIDJSON_BUILD_EXAMPLES OFF"
              "RAPIDJSON_BUILD_TESTS OFF"
              "RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF"
    )

    if(TARGET rapidjson AND NOT TARGET rapidjson::rapidjson)
        add_library(rapidjson::rapidjson ALIAS rapidjson)
    endif()

    rapids_export_package(BUILD rapidjson rapids_triton-exports)
    rapids_export_package(INSTALL rapidjson rapids_triton-exports)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD rapidjson [=[${CMAKE_CURRENT_LIST_DIR}]=] rapids_triton-exports)

endfunction()

find_and_configure_rapidjson("1.1.0")