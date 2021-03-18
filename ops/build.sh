#!/bin/bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
if [ ! -d "${BUILD_DIR}" ]
then
  mkdir ${BUILD_DIR}
fi
pushd "$BUILD_DIR" > /dev/null 2>&1
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH="${PWD}/install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    .. \
  && make install
status=$?
popd > /dev/null 2>&1
exit $status
