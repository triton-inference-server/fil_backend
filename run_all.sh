#!/bin/bash
set -e

docker build -t triton_fil -f ops/Dockerfile .
docker build -t triton_test -f qa/Dockerfile .
gpus=${CUDA_VISIBLE_DEVICES:-all}
docker run --rm --gpus $gpus triton_test
