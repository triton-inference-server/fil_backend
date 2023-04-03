#!/bin/bash

set -e

# ENVIRONMENT VARIABLE OPTIONS
# RETRAIN: 1 to force retraining of existing models, 0 to use existing models
#   if available
# USE_CLIENT_WHEEL: 1 to install Triton client from wheel for tests
# SDK_IMAGE: If set, copy client wheel from this SDK image
# HOST_BUILD: 1 to build backend lib on host and use it in tests

REPO_DIR=$(cd $(dirname $0)/../../; pwd)
QA_DIR="${REPO_DIR}/qa"
MODEL_DIR="${QA_DIR}/L0_e2e/model_repository"
MODEL_CACHE_DIR="${QA_DIR}/L0_e2e/.model_cache"
HOST_BUILD="${HOST_BUILD:-0}"
TEST_PROFILE="${TEST_PROFILE:-dev}"

export SERVER_TAG=triton_fil
export TEST_TAG=triton_fil_test

if [ ! -z $SDK_IMAGE ]
then
  export SDK_IMAGE="${SDK_IMAGE}"
  USE_CLIENT_WHEEL=1
fi
if [ ! -z $USE_CLIENT_WHEEL ]
then
  export USE_CLIENT_WHEEL="${USE_CLIENT_WHEEL}"
fi

BUILD_ARGS=''
if [ $HOST_BUILD -eq 1 ]
then
  BUILD_ARGS="$BUILD_ARGS --host"
fi

echo "Building Docker images..."
$REPO_DIR/build.sh $BUILD_ARGS

DOCKER_ARGS="-t -v ${QA_DIR}/logs:/qa/logs"

if [ -z $CUDA_VISIBLE_DEVICES ]
then
  DOCKER_ARGS="$DOCKER_ARGS --gpus all"
else
  DOCKER_ARGS="$DOCKER_ARGS --gpus \"device=$CUDA_VISIBLE_DEVICES\""
fi

echo "Running GPU-enabled tests..."
mkdir -p "${MODEL_CACHE_DIR}"
docker run \
  $DOCKER_ARGS \
  -e TEST_PROFILE=$TEST_PROFILE \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${MODEL_CACHE_DIR}:/qa/L0_e2e/.model_cache" \
  --rm $TEST_TAG

export SERVER_TAG=triton_fil:cpu
export TEST_TAG=triton_fil_test:cpu

echo "Building CPU-only Docker images..."
$REPO_DIR/build.sh $BUILD_ARGS --cpu-only

echo "Running CPU-only tests..."
docker run \
  $DOCKER_ARGS \
  -e TRITON_ENABLE_GPU=OFF \
  -e TEST_PROFILE=$TEST_PROFILE \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${MODEL_CACHE_DIR}:/qa/L0_e2e/.model_cache" \
  --rm $TEST_TAG
