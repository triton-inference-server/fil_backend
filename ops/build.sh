#!/bin/bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
pushd "$BUILD_DIR" > /dev/null 2>&1
cmake -DCMAKE_INSTALL_PREFIX:PATH="${PWD}/install" ..
make install
popd > /dev/null 2>&1
