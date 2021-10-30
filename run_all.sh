#!/bin/bash
set -e

docker build --no-cache -t triton_fil -f ops/Dockerfile .
docker build -t triton_test -f qa/Dockerfile .
gpus=${CUDA_VISIBLE_DEVICES:-all}
docker run -v $PWD/logs:/logs --name triton_test --rm --gpus $gpus triton_test