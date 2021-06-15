#!/bin/bash

PARALLEL=${PARALLEL:=$(nproc)}
BASE_IMAGE=${BASE_IMAGE:=ubuntu:20.04}
TAG=${TAG:=triton_fil}

FIL=${FIL:=1}

DOCKER_BUILDKIT=1 docker build \
  -t "$TAG" \
  -f ops/e2e.Dockerfile \
  --build-arg PARALLEL="$PARALLEL" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg FIL="$FIL" \
  .
