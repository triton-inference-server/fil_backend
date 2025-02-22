# syntax = docker/dockerfile:1.3
###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Version of Triton to use
ARG TRITON_VERSION=25.01
# Base container image
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
# Whether or not to enable GPU build
ARG TRITON_ENABLE_GPU=ON
# A Triton server image to use as base for test layers (skip actual build)
ARG SERVER_IMAGE=build-stage
# Whether or not to install Triton client from wheel in SDK image
ARG USE_CLIENT_WHEEL=0
# SDK container image (only used if USE_CLIENT_WHEEL==1)
ARG SDK_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk
# Whether or not to use backend library prebuilt on host
ARG USE_HOST_LIB=0

FROM condaforge/miniforge3 as conda-base
COPY ./ops/gpuci_conda_retry /usr/bin/gpuci_conda_retry
COPY ./ops/gpuci_mamba_retry /usr/bin/gpuci_mamba_retry
RUN chmod +x /usr/bin/gpuci_conda_retry /usr/bin/gpuci_mamba_retry

RUN mkdir /conda
RUN gpuci_mamba_retry install -c conda-forge conda-pack=0.7

FROM conda-base as conda-dev
COPY ./conda/environments/rapids_triton_dev.yml /conda/environment.yml
RUN gpuci_mamba_retry env update -f /conda/environment.yml \
 && rm /conda/environment.yml
RUN conda-pack -n rapids_triton_dev -o /tmp/env.tar \
 && mkdir /conda/dev/ \
 && cd /conda/dev/ \
 && tar xf /tmp/env.tar \
 && rm /tmp/env.tar
RUN /conda/dev/bin/conda-unpack

# Stage for installing test dependencies
FROM conda-base as base-test-install
COPY ./conda/environments/triton_test_no_client.yml /environment.yml

RUN gpuci_mamba_retry env update -f /environment.yml \
    && rm /environment.yml

FROM base-test-install as wheel-install-0
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      build-essential \
      ca-certificates \
      git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && conda run --no-capture-output -n triton_test pip install tritonclient[all]

FROM ${SDK_IMAGE} as sdk-image

FROM base-test-install as wheel-install-1
COPY --from=sdk-image /workspace/install/python /sdk_install
RUN conda run --no-capture-output -n triton_test \
    pip install /sdk_install/tritonclient*manylinux*.whl \
 && rm -r /sdk_install

FROM wheel-install-${USE_CLIENT_WHEEL} as conda-test
RUN conda run --no-capture-output -n triton_test \
    pip install git+https://github.com/rapidsai/rapids-triton.git@branch-24.10#subdirectory=python
RUN conda-pack --ignore-missing-files -n triton_test -o /tmp/env.tar \
 && mkdir /conda/test/ \
 && cd /conda/test/ \
 && tar xf /tmp/env.tar \
 && rm /tmp/env.tar
RUN /conda/test/bin/conda-unpack


FROM ${BASE_IMAGE} as base

ENV PATH="/root/miniconda3/bin:${PATH}"

# In CI, CPU base image may not have curl, but it also does not need to update
# the cuda keys
RUN  if command -v curl; \
 then [ $(uname -m) = 'x86_64' ] \
 && curl -L -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
 || curl -L -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-keyring_1.0-1_all.deb; \
 dpkg -i /tmp/cuda-keyring.deb \
 && rm /tmp/cuda-keyring.deb; fi

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      build-essential \
      ca-certificates \
      git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage immediately before building; useful for build iteration
FROM base as build-prep

RUN mkdir -p /rapids_triton/build /rapids_triton/install

COPY ./src /rapids_triton/src
COPY ./CMakeLists.txt /rapids_triton
COPY ./cmake /rapids_triton/cmake

ARG BACKEND_NAME=fil
ENV BACKEND_NAME=$BACKEND_NAME

WORKDIR /rapids_triton/build

# Remove potentially stale build artifacts
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    else \
      mkdir -p /opt/tritonserver/backends/${BACKEND_NAME}; \
    fi

# Stage where build actually takes place
FROM build-prep as build-stage

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BUILD_TYPE=Release
ENV BUILD_TYPE=$BUILD_TYPE
ARG BUILD_TESTS
ENV BUILD_TESTS=$BUILD_TESTS
ARG BUILD_EXAMPLE
ENV BUILD_EXAMPLE=$BUILD_EXAMPLE

ARG TRITON_REPO_ORGANIZATION=https://github.com/triton-inference-server
ENV TRITON_REPO_ORGANIZATION=$TRITON_REPO_ORGANIZATION
ARG TRITON_CORE_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG
ARG TRITON_COMMON_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_COMMON_REPO_TAG=$TRITON_COMMON_REPO_TAG
ARG TRITON_BACKEND_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_BACKEND_REPO_TAG=$TRITON_BACKEND_REPO_TAG
ARG RAPIDS_TRITON_REPO_TAG=main
ENV RAPIDS_TRITON_REPO_TAG=$RAPIDS_TRITON_REPO_TAG
ARG RAPIDS_TRITON_REPO_PATH=https://github.com/rapidsai/rapids-triton.git
ENV RAPIDS_TRITON_REPO_PATH=$RAPIDS_TRITON_REPO_PATH

ARG TRITON_ENABLE_GPU=ON
ENV TRITON_ENABLE_GPU=$TRITON_ENABLE_GPU
ARG TRITON_ENABLE_STATS=ON
ENV TRITON_ENABLE_GPU=$TRITON_ENABLE_GPU

# Specify *minimum* version for all RAPIDS dependencies
# Some RAPIDS deps may have later versions
ARG RAPIDS_DEPENDENCIES_VERSION=24.10
ENV RAPIDS_DEPENDENCIES_VERSION=$RAPIDS_DEPENDENCIES_VERSION

ARG TRITON_FIL_USE_TREELITE_STATIC=ON
ENV TRITON_FIL_USE_TREELITE_STATIC=$TRITON_FIL_USE_TREELITE_STATIC

COPY --from=conda-dev /conda/dev /conda/dev

SHELL ["/bin/bash", "-c"]

RUN source /conda/dev/bin/activate \
 && cmake \
      --log-level=VERBOSE \
      -GNinja \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DBUILD_TESTS="${BUILD_TESTS}" \
      -DTRITON_REPO_ORGANIZATION="${TRITON_REPO_ORGANIZATION}" \
      -DTRITON_CORE_REPO_TAG="${TRITON_CORE_REPO_TAG}" \
      -DTRITON_COMMON_REPO_TAG="${TRITON_COMMON_REPO_TAG}" \
      -DTRITON_BACKEND_REPO_TAG="${TRITON_BACKEND_REPO_TAG}" \
      -DRAPIDS_TRITON_REPO_TAG="${RAPIDS_TRITON_REPO_TAG}" \
      -DRAPIDS_TRITON_REPO_PATH="${RAPIDS_TRITON_REPO_PATH}" \
      -DTRITON_ENABLE_GPU="${TRITON_ENABLE_GPU}" \
      -DTRITON_ENABLE_STATS="${TRITON_ENABLE_STATS}" \
      -DRAPIDS_DEPENDENCIES_VERSION="${RAPIDS_DEPENDENCIES_VERSION}" \
      -DTRITON_FIL_USE_TREELITE_STATIC="${TRITON_FIL_USE_TREELITE_STATIC}" \
      -DCMAKE_INSTALL_PREFIX=/rapids_triton/install \
      ..;

ENV CCACHE_DIR=/ccache

RUN --mount=type=cache,target=/ccache/ source /conda/dev/bin/activate && ninja install

ARG CCACHE_REMOTE_STORAGE
RUN if [ -n "${CCACHE_REMOTE_STORAGE}" ] ; then \
      apt-get update -qq && apt-get install -y ccache ; \
      ccache --set-config=remote_only=true ; \
      ccache --set-config=remote_storage=${CCACHE_REMOTE_STORAGE} ; \
      ccache --set-config=log_file=/tmp/ccache.log ; \
      ccache -p ; \
      export CMAKE_C_COMPILER_LAUNCHER=ccache ; \
      export CMAKE_CXX_COMPILER_LAUNCHER=ccache ; \
      export CMAKE_CUDA_COMPILER_LAUNCHER=ccache ; \
      export VERBOSE=1 ; \
    fi

# Stage for generating testing image
FROM ${SERVER_IMAGE} as test-host-0
FROM ${SERVER_IMAGE} as test-host-1

ARG BACKEND_NAME=fil
ENV BACKEND_NAME=$BACKEND_NAME

# Remove existing FIL backend install
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    fi
COPY ./install/backends/fil /opt/tritonserver/backends/${BACKEND_NAME}

FROM test-host-${USE_HOST_LIB} as test-build

FROM ${SERVER_IMAGE} as test-stage
ARG BACKEND_NAME=fil
ENV BACKEND_NAME=$BACKEND_NAME

COPY --from=conda-test /conda/test /conda/test

# Remove existing FIL backend install
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    fi
COPY --from=test-build \
  /opt/tritonserver/backends/$BACKEND_NAME \
  /opt/tritonserver/backends/$BACKEND_NAME

COPY qa /qa
COPY scripts /scripts

ENTRYPOINT []
CMD ["/bin/bash", "-c", "source /conda/test/bin/activate && /qa/entrypoint.sh"]

FROM ${BASE_IMAGE} as final

ARG BACKEND_NAME=fil
ENV BACKEND_NAME=$BACKEND_NAME

RUN mkdir /models

# Remove existing FIL backend install
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    fi

COPY --from=build-stage \
  /opt/tritonserver/backends/$BACKEND_NAME \
  /opt/tritonserver/backends/$BACKEND_NAME
