#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR="${SCRIPT_DIR}/../build"

if [ $# -ne 3 ]
then
  echo "usage: $0 model_file data_file max_batch_size"
  exit 1
fi

model_file=$(readlink -f $1)
data_file=$(readlink -f $2)
max_batch="$3"

if [ ! -d "$BUILD_DIR" ]
then
  mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

cmake \
  --log-level=VERBOSE \
  -GNinja \
  -DCMAKE_BUILD_TYPE="Release" \
  -DBUILD_MICROBENCHMARK=ON \
  -DBUILD_TESTS="OFF" \
  -DTRITON_CORE_REPO_TAG="r22.08" \
  -DTRITON_COMMON_REPO_TAG="r22.08" \
  -DTRITON_BACKEND_REPO_TAG="r22.08" \
  -DRAPIDS_DEPENDENCIES_VERSION="22.08" \
  -DTRITON_FIL_USE_TREELITE_STATIC="ON" \
  -DTRITON_FIL_ENABLE_TREESHAP="OFF" \
  -DTRITON_ENABLE_GPU="ON" \
  ..;

ninja forest_bench

echo "./forest_bench $model_file $data_file -b $max_batch"
./forest_bench \
  "$model_file" \
  "$data_file" \
  -b 1 16 128 1024 65536 "$max_batch" \
  -a fil_sparse fil_sparse8 fil_dense fil_dense_reorg herring_gpu herring_cpu
