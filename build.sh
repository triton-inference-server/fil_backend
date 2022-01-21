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
VALIDTARGETS="server tests"
VALIDFLAGS="--cpu-only -g -h --help"
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   server           - build a Triton server container with FIL backend
   tests            - build container(s) with unit tests
 and <flag> is:
   -g               - build for debug
   -h               - print this text
   --cpu-only       - build CPU-only versions of targets
   --tag-commit     - tag Docker images based on current git commit
   --buildpy        - use Triton's build.py script for build
   --no-cache       - disable Docker cache for build

 default action (no args) is to build all targets
 The following environment variables are also accepted to allow further customization:
   BASE_IMAGE       - Base image for Docker images, or build image for build.py
   TRITON_VERSION   - Triton version to use for build
   SERVER_TAG       - The tag to use for the server image
   TEST_TAG         - The tag to use for the test image
   PREBUILT_IMAGE   - A server image to be tested (used as base of test image)
   TRITON_REF       - Commit ref for Triton when using build.py
   COMMON_REF       - Commit ref for Triton common repo when using build.py
   CORE_REF         - Commit ref for Triton core repo when using build.py
   BACKEND_REF      - Commit ref for Triton backend repo when using build.py
   THIRDPARTY_REF   - Commit ref for Triton third-party repos when using build.py
   JOB_ID           - A unique id to use for this build job
   USE_CLIENT_WHEEL - If 1, Triton client will be installed from wheel
   SDK_IMAGE        - If set, client wheel will be copied from this image
"

BUILD_TYPE=Release
TRITON_ENABLE_GPU=ON
DOCKER_ARGS=""
BUILDPY=0

export DOCKER_BUILDKIT=1

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function completeBuild {
    (( ${NUMARGS} == 0 )) && return
    for a in ${ARGS}; do
        if (echo " ${VALIDTARGETS} " | grep -q " ${a} "); then
          false; return
        fi
    done
    true
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Long arguments
LONG_ARGUMENT_LIST=(
    "cpu-only"
    "tag-commit"
    "buildpy"
    "no-cache"
)

# Short arguments
ARGUMENT_LIST=(
    "g"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s," "${LONG_ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "$(printf "%s" "${ARGUMENT_LIST[@]}")" \
    -- "$@"
)

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$opts"

while true
do
  case "$1" in
    -g | --debug )
      BUILD_TYPE=Debug
      ;;
    --cpu-only )
      TRITON_ENABLE_GPU=OFF
      ;;
    --tag-commit )
      [ -z $SERVER_TAG ] \
        && SERVER_TAG="triton_fil:$(cd $REPODIR; git rev-parse --short HEAD)" \
        || true
      [ -z $TEST_TAG ] \
        && TEST_TAG="triton_fil_test:$(cd $REPODIR; git rev-parse --short HEAD)" \
        || true
      ;;
    --buildpy )
      BUILDPY=1
      ;;
    --no-cache )
      DOCKER_ARGS="$DOCKER_ARGS --no-cache"
      ;;
    --)
      shift
      break
      ;;
  esac
  shift
done

if [ -z $SERVER_TAG ]
then
  SERVER_TAG='triton_fil'
fi
if [ -z $TEST_TAG ]
then
  TEST_TAG='triton_fil_test'
fi

DOCKER_ARGS="$DOCKER_ARGS --build-arg BUILD_TYPE=${BUILD_TYPE}"
DOCKER_ARGS="$DOCKER_ARGS --build-arg TRITON_ENABLE_GPU=${TRITON_ENABLE_GPU}"

if [ -z $BASE_IMAGE ]
then
  BUILDPY_OPT=''
else
  DOCKER_ARGS="$DOCKER_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
  BUILDPY_OPT="--image=base,$BASE_IMAGE"
fi

if [ $TRITON_ENABLE_GPU != 'OFF' ]
then
  BUILDPY_OPT="${BUILDPY_OPT} --enable-gpu"
fi

if [ ! -z $TRITON_VERSION ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg TRITON_VERSION=${TRITON_VERSION}"
fi

if [ ! -z $SDK_IMAGE ]
then
  USE_CLIENT_WHEEL=1
  DOCKER_ARGS="$DOCKER_ARGS --build-arg SDK_IMAGE=${SDK_IMAGE}"
fi

if [ ! -z $USE_CLIENT_WHEEL ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg USE_CLIENT_WHEEL=${USE_CLIENT_WHEEL}"
fi

if completeBuild || hasArg server
then
  BACKEND=1
fi

if completeBuild || hasArg tests
then
  TESTS=1
  DOCKER_ARGS="$DOCKER_ARGS --build-arg BUILD_TESTS=ON"
fi

buildpy () {
  pushd "$REPODIR"
  branch=$(git rev-parse --abbrev-ref HEAD) || branch='HEAD'
  if [ $branch = 'HEAD' ]
  then
    branch='main'
  fi
  echo "build.sh: Building on branch '$branch' with build.py"

  if [ -z $JOB_ID ]
  then
    build_dir="$REPODIR/build/"
  else
    build_dir="$REPODIR/build-$JOB_ID/"
  fi
  mkdir -p "$build_dir"

  TRITON_REF=${TRITON_REF:-main}
  COMMON_REF=${COMMON_REF:-main}
  CORE_REF=${CORE_REF:-main}
  BACKEND_REF=${BACKEND_REF:-main}
  THIRDPARTY_REF=${THIRDPARTY_REF:-main}

  server_repo="${build_dir}/triton_server"
  if [ -d "$server_repo" ]
  then
    rm -rf "$server_repo"
  fi
  git clone https://github.com/triton-inference-server/server.git \
    "${server_repo}" \
    -b $TRITON_REF \
    --depth 1

  pushd "${server_repo}"
  python3 build.py $BUILDPY_OPT \
    --cmake-dir=./build \
    --build-dir=/tmp/citritonbuild \
    --enable-logging \
    --enable-metrics \
    --enable-stats \
    --endpoint=http \
    --endpoint=grpc \
    --repo-tag=common:$COMMON_REF \
    --repo-tag=core:$CORE_REF \
    --repo-tag=backend:$BACKEND_REF \
    --repo-tag=thirdparty:$THIRDPARTY_REF \
    --backend=fil:$branch
  docker tag tritonserver:latest $SERVER_TAG

  popd
  popd

}

if [ $BACKEND -eq 1 ]
then
  if [ $BUILDPY -eq 1 ]
  then
    buildpy
  else
    docker build \
      $DOCKER_ARGS \
      -t "$SERVER_TAG" \
      -f ops/Dockerfile \
      $REPODIR
  fi
fi

if [ ! -z $PREBUILT_IMAGE ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg SERVER_IMAGE=${PREBUILT_IMAGE}"
elif [ $BUILDPY -eq 1 ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg SERVER_IMAGE=${SERVER_TAG}"
fi

if [ $TESTS -eq 1 ]
then
  docker build \
    $DOCKER_ARGS \
    --target test-stage \
    -t "$TEST_TAG" \
    -f ops/Dockerfile \
    $REPODIR
fi
