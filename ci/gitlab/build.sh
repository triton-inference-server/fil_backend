#!/bin/bash

set -ex

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
  LOG_DIR="qa/logs"
else
  LOG_DIR="$(readlink -f $LOG_DIR)"
fi
if [ ! -d "${LOG_DIR}" ]
then
  mkdir -p "${LOG_DIR}"
fi

if [ -z "$NV_DOCKER_ARGS" ]
then
  if [ -z $CUDA_VISIBLE_DEVICES ]
  then
    GPU_DOCKER_ARGS='--gpus all'
  else
    GPU_DOCKER_ARGS='--gpus $CUDA_VISIBLE_DEVICES'
  fi
else
  GPU_DOCKER_ARGS="$(eval ${NV_DOCKER_ARGS} || echo -n '')"
fi

if [ ! -z $RUNNER_ID ]
then
  DOCKER_ARGS="$DOCKER_ARGS --label RUNNER_ID=${RUNNER_ID}"
fi

echo "Generating example models..."
# Use 'docker cp' instead of mounting, because we cannot mount directories
# from the GitLab runner due to the "Docker-outside-of-Docker" architecture.
# See https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DL&title=GitLab+Runner
# for more details.
MODEL_BUILDER_INST=model_builder_inst_$(uuidgen | sed -e 's/-//g')
docker create -t --name ${MODEL_BUILDER_INST} \
  -e RETRAIN=1 \
  -e OWNER_ID=$(id -u) \
  -e OWNER_GID=$(id -g) \
  $GPU_DOCKER_ARGS \
  $DOCKER_ARGS \
  $MODEL_BUILDER_IMAGE
docker start ${MODEL_BUILDER_INST}
docker exec ${MODEL_BUILDER_INST} bash -c 'mkdir -p /qa/L0_e2e/ && mkdir -p /qa/logs/'
mkdir -p qa/L0_e2e/model_repository/
mkdir -p qa/L0_e2e/cpu_model_repository/
docker cp qa/L0_e2e/model_repository/ ${MODEL_BUILDER_INST}:/qa/L0_e2e/
docker cp qa/L0_e2e/cpu_model_repository/ ${MODEL_BUILDER_INST}:/qa/L0_e2e/
docker exec ${MODEL_BUILDER_INST} bash -c 'find /qa/'

docker exec \
  ${MODEL_BUILDER_INST} \
  bash -c 'source /conda/test/bin/activate && /qa/generate_example_models.sh'

docker cp ${MODEL_BUILDER_INST}:/qa/L0_e2e/model_repository/ qa/L0_e2e/
docker cp ${MODEL_BUILDER_INST}:/qa/L0_e2e/cpu_model_repository/ qa/L0_e2e/
docker cp ${MODEL_BUILDER_INST}:/qa/logs/. "${LOG_DIR}"
docker stop ${MODEL_BUILDER_INST}
docker rm ${MODEL_BUILDER_INST}

find "${LOG_DIR}"
find qa/L0_e2e/model_repository/
find qa/L0_e2e/cpu_model_repository/

if [ $CPU_ONLY -eq 1 ]
then
  DOCKER_ARGS="${DOCKER_ARGS} -e TRITON_ENABLE_GPU=OFF"
else
  DOCKER_ARGS="${DOCKER_ARGS} ${GPU_DOCKER_ARGS}"
fi

echo "Running tests..."
TEST_INST=test_inst_$(uuidgen | sed -e 's/-//g')
docker create -t --name ${TEST_INST} \
  -e TEST_PROFILE=ci \
  $DOCKER_ARGS \
  $TEST_TAG
docker start ${TEST_INST}
docker exec ${TEST_INST} bash -c 'mkdir -p /qa/L0_e2e/ && mkdir -p /qa/logs/'
docker cp qa/L0_e2e/model_repository/ ${TEST_INST}:/qa/L0_e2e/
docker cp qa/L0_e2e/cpu_model_repository/ ${TEST_INST}:/qa/L0_e2e/
docker exec ${TEST_INST} bash -c 'find /qa/'
docker exec ${TEST_INST}

docker cp ${TEST_INST}:/qa/logs/. "${LOG_DIR}"
docker stop ${TEST_INST}
docker rm ${TEST_INST}
