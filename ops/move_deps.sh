#!/bin/bash
set -e
FIL_LIB="${FIL_LIB:-libtriton_fil.so}"
LIB_DIR="${LIB_DIR:-/usr/lib}"
if [ -z "${CONDA_LIB_DIR}" ] && [ -z "${CONDA_PREFIX}" ]
then
  echo "Must set CONDA_LIB_DIR to conda environment lib directory"
  exit 1
fi
CONDA_LIB_DIR="${CONDA_LIB_DIR:-$CONDA_PREFIX/lib}"

if [ ! -d "${LIB_DIR}" ]
then
  mkdir -p "${LIB_DIR}"
fi

deps_list=$(ldd "${FIL_LIB}" | grep "${CONDA_LIB_DIR}" | awk '{print $3}' | grep -v libcudart\.so)
cp $deps_list "${LIB_DIR}"
