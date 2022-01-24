#!/bin/bash

set -e

# ENVIRONMENT VARIABLE OPTIONS
# RETRAIN: 1 to force retraining of existing models, 0 to use existing models
#   if available

REPO_DIR=$(cd $(dirname $0)/../../; pwd)
QA_DIR="${REPO_DIR}/qa"
MODEL_DIR="${QA_DIR}/L0_e2e/model_repository"
CPU_MODEL_DIR="${QA_DIR}/L0_e2e/cpu_model_repository"

export SERVER_TAG=triton_fil
export TEST_TAG=triton_fil_test

echo "Building Docker images..."
$REPO_DIR/build.sh

DOCKER_ARGS="-t -v ${QA_DIR}/logs:/qa/logs"

if [ -z $CUDA_VISIBLE_DEVICES ]
then
  DOCKER_ARGS="$DOCKER_ARGS --gpus all"
else
  DOCKER_ARGS="$DOCKER_ARGS --gpus $CUDA_VISIBLE_DEVICES"
fi

echo "Running linters..."
docker run \
  $DOCKER_ARGS \
  -v "${REPO_DIR}:/rapids_triton_src" \
  --rm $TEST_TAG \
  bash -c 'cd /rapids_triton_src && conda run -n triton_test python qa/run-clang-format.py && conda run -n triton_test flake8'

echo "Generating example models..."
docker run \
  -e RETRAIN=${RETRAIN:-0} \
  -e OWNER_ID=$(id -u) \
  -e OWNER_GID=$(id -g) \
  $DOCKER_ARGS \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${CPU_MODEL_DIR}:/qa/L0_e2e/cpu_model_repository" \
  --rm $TEST_TAG \
  bash -c 'conda run -n triton_test /qa/generate_example_models.sh'

echo "Running GPU-enabled tests..."
docker run \
  $DOCKER_ARGS \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${CPU_MODEL_DIR}:/qa/L0_e2e/cpu_model_repository" \
  --rm $TEST_TAG

export SERVER_TAG=triton_fil:cpu
export TEST_TAG=triton_fil_test:cpu

echo "Building CPU-only Docker images..."
$REPO_DIR/build.sh --cpu-only

echo "Running CPU-only tests..."
docker run \
  $DOCKER_ARGS \
  -v "${MODEL_DIR}:/qa/L0_e2e/model_repository" \
  -v "${CPU_MODEL_DIR}:/qa/L0_e2e/cpu_model_repository" \
  --rm $TEST_TAG
