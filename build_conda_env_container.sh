#!/bin/bash
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

set -e

REPODIR=$(cd $(dirname $0); pwd)

NUMARGS=$#
ARGS=$*
VALIDTARGETS="conda-dev conda-test"
VALIDFLAGS="-h --help"
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 <target> [<flag> ...]
 where <target> is:
   conda-dev        - build container with dev Conda env
   conda-test       - build container with test Conda env
 and <flag> is:
   -h               - print this text

 The following environment variables are also accepted to allow further customization:
   CONDA_DEV_TAG    - The tag to use for the image containing dev Conda env
   CONDA_TEST_TAG   - The tag to use for the image containing test Conda env
"

export DOCKER_BUILDKIT=1

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help || (( ${NUMARGS} == 0 ))
then
    echo "${HELP}"
    exit 0
fi

if [ -z $CONDA_DEV_TAG ]
then
  CONDA_DEV_TAG='triton_fil_dev_conda'
fi
if [ -z $CONDA_TEST_TAG ]
then
  CONDA_TEST_TAG='triton_fil_test_conda'
fi

BUILD_CONDA_DEV=0
BUILD_CONDA_TEST=0
if hasArg conda-dev
then
  BUILD_CONDA_DEV=1
elif hasArg conda-test
then
  BUILD_CONDA_TEST=1
fi

if [ $BUILD_CONDA_DEV -eq 1 ]
then
  docker build \
    $DOCKER_ARGS \
    --target conda-dev \
    -t "$CONDA_DEV_TAG" \
    -f ops/Dockerfile \
    $REPODIR
fi

if [ $BUILD_CONDA_TEST -eq 1 ]
then
  docker build \
    $DOCKER_ARGS \
    --target base-test-install \
    -t "$CONDA_TEST_TAG" \
    -f ops/Dockerfile \
    $REPODIR
fi
