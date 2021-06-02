#!/bin/bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TRITON_VERSION="${TRITON_VERSION:-main}"

CALVER_RE='^[0-9]+[.][0-9]+$'

if [[ "${TRITON_VERSION}" =~ $CALVER_RE ]]
then
  TRITON_VERSION="r${TRITON_VERSION}"
fi

if [ ! -d "${BUILD_DIR}" ]
then
  mkdir ${BUILD_DIR}
fi
pushd "$BUILD_DIR" > /dev/null 2>&1
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH="${PWD}/install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DTRITON_BACKEND_REPO_TAG="${TRITON_VERSION}" \
    -DTRITON_COMMON_REPO_TAG="${TRITON_VERSION}" \
    -DTRITON_CORE_REPO_TAG="${TRITON_VERSION}" \
    .. \
  && make install
status=$?
popd > /dev/null 2>&1
exit $status
