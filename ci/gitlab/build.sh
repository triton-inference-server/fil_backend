#!/bin/bash

set -e

# ENVIRONMENT VARIABLE OPTIONS
# PREBUILT_SERVER_TAG: The tag of the prebuilt Triton server image to test
# PREBUILT_TEST_TAG: The tag of the prebuilt test image to run tests in
# MODEL_BUILDER_IMAGE: A Docker image to be used for training test models
# LOG_DIR: Host directory for storing logs
# NV_DOCKER_ARGS: A bash expression that (when evaluated) returns Docker
#   arguments for controlling GPU access
# BUILDPY: 1 to use Triton's build.py script for server build
# CPU_ONLY: 1 to build without GPU support
# NO_CACHE: 0 to enable Docker cache during build
# USE_CLIENT_WHEEL: 1 to install Triton client from wheel for tests
# SDK_IMAGE: If set, copy client wheel from this SDK image

REPO_DIR=$(cd $(dirname $0)/../../; pwd)
QA_DIR="${REPO_DIR}/qa"
MODEL_DIR="${QA_DIR}/L0_e2e/model_repository"
MODEL_CACHE_DIR="${QA_DIR}/L0_e2e/.model_cache"
BUILDPY=${BUILDPY:-0}
CPU_ONLY=${CPU_ONLY:-0}
NO_CACHE=${NO_CACHE:-1}

if [ -z $CI_COMMIT_BRANCH ]
then
  export BUILDPY_BRANCH="$CI_COMMIT_BRANCH"
fi

# Check if test or base images need to be built and do so if necessary
if [ -z $PREBUILT_SERVER_TAG ]
then
  export SERVER_TAG=triton_fil
else
  export PREBUILT_IMAGE="$PREBUILT_SERVER_TAG"
  export SERVER_TAG="$PREBUILT_SERVER_TAG"
fi
[ -z $TRITON_SERVER_REPO_TAG ] || export TRITON_REF="$TRITON_SERVER_REPO_TAG"
[ -z $TRITON_COMMON_REPO_TAG ] || export COMMON_REF="$TRITON_COMMON_REPO_TAG"
[ -z $TRITON_CORE_REPO_TAG ] || export CORE_REF="$TRITON_CORE_REPO_TAG"
[ -z $TRITON_BACKEND_REPO_TAG ] || export BACKEND_REF="$TRITON_BACKEND_REPO_TAG"

if [ -z $PREBUILT_TEST_TAG ]
then
  export TEST_TAG=triton_fil_test
  echo "Building Docker images..."
  if [ $BUILDPY -eq 1 ]
  then
    BUILDARGS='--buildpy'
  else
    BUILDARGS=''
  fi
  if [ $CPU_ONLY -eq 1 ]
  then
    BUILDARGS="$BUILDARGS --cpu-only"
  fi
  if [ $NO_CACHE -eq 1 ]
  then
    BUILDARGS="$BUILDARGS --no-cache"
  fi
  if [ ! -z $SDK_IMAGE ]
  then
    USE_CLIENT_WHEEL=1
    export SDK_IMAGE="${SDK_IMAGE}"
  fi
  if [ ! -z $USE_CLIENT_WHEEL ]
  then
    export USE_CLIENT_WHEEL="${USE_CLIENT_WHEEL}"
  fi
  $REPO_DIR/build.sh $BUILDARGS
else
  export TEST_TAG="$PREBUILT_TEST_TAG"
fi

MODEL_BUILDER_IMAGE=${MODEL_BUILDER_IMAGE:-${TEST_TAG}}

# Set up directory for logging
if [ -z $LOG_DIR ]
then
  LOG_DIR="${QA_DIR}/logs"
fi
if [ ! -d "${LOG_DIR}" ]
then
  mkdir -p "${LOG_DIR}"
fi
LOG_DIR="$(readlink -f $LOG_DIR)"

DOCKER_ARGS="-v ${LOG_DIR}:/qa/logs"

if [ -z "$NV_DOCKER_ARGS" ]
then
  if [ -z $CUDA_VISIBLE_DEVICES ]
  then
    GPU_DOCKER_ARGS='--gpus all'
  else
    GPU_DOCKER_ARGS='--gpus $CUDA_VISIBLE_DEVICES'
  fi
else
  GPU_DOCKER_ARGS="$(eval ${NV_DOCKER_ARGS})"
fi

if [ ! -z $RUNNER_ID ]
then
  DOCKER_ARGS="$DOCKER_ARGS --label RUNNER_ID=${RUNNER_ID}"
fi

if [ $CPU_ONLY -eq 1 ]
then
  DOCKER_ARGS="${DOCKER_ARGS} -e TRITON_ENABLE_GPU=OFF"
else
  DOCKER_ARGS="${DOCKER_ARGS} ${GPU_DOCKER_ARGS}"
fi

echo "Running tests..."
mkdir -p "${MODEL_CACHE_DIR}"
docker run \
  -e TEST_PROFILE=ci \
  $DOCKER_ARGS \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${MODEL_CACHE_DIR}:/qa/L0_e2e/.model_cache" \
  --rm $TEST_TAG
