# syntax = docker/dockerfile:1.3
###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Version of Triton to use
ARG TRITON_VERSION=22.02
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

FROM ${BASE_IMAGE} as base

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=true

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-$(uname -m).sh -b \
    && rm -f Miniconda3-latest-Linux-$(uname -m).sh 

COPY ./conda/environments/rapids_triton_dev.yml /environment.yml

RUN --mount=type=cache,target=/root/miniconda3/pkgs conda env update -f /environment.yml \
    && rm /environment.yml

RUN conda init && echo 'conda activate rapids_triton_dev' >> /root/.bashrc

SHELL ["conda", "run", "--no-capture-output", "-n", "rapids_triton_dev", "/bin/bash", "-c"]

ENV PYTHONDONTWRITEBYTECODE=false

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
      mkdir /opt/tritonserver/backends/${BACKEND_NAME}; \
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

ARG TRITON_CORE_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_CORE_REPO_TAG=$TRITON_CORE_REPO_TAG
ARG TRITON_COMMON_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_COMMON_REPO_TAG=$TRITON_COMMON_REPO_TAG
ARG TRITON_BACKEND_REPO_TAG=r${TRITON_VERSION}
ENV TRITON_BACKEND_REPO_TAG=$TRITON_BACKEND_REPO_TAG

ARG TRITON_ENABLE_GPU=ON
ENV TRITON_ENABLE_GPU=$TRITON_ENABLE_GPU
ARG TRITON_ENABLE_STATS=ON
ENV TRITON_ENABLE_GPU=$TRITON_ENABLE_GPU

ARG RAPIDS_DEPENDENCIES_VERSON=22.04
ENV RAPIDS_DEPENDENCIES_VERSON=$RAPIDS_DEPENDENCIES_VERSON

ARG TRITON_FIL_USE_TREELITE_STATIC=ON
ENV TRITON_FIL_USE_TREELITE_STATIC=$TRITON_FIL_USE_TREELITE_STATIC

ARG TRITON_FIL_ENABLE_TREESHAP=ON
ENV TRITON_FIL_ENABLE_TREESHAP=$TRITON_FIL_ENABLE_TREESHAP

RUN cmake \
    --log-level=VERBOSE \
    -GNinja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS="${BUILD_TESTS}" \
    -DTRITON_CORE_REPO_TAG="${TRITON_CORE_REPO_TAG}" \
    -DTRITON_COMMON_REPO_TAG="${TRITON_COMMON_REPO_TAG}" \
    -DTRITON_BACKEND_REPO_TAG="${TRITON_BACKEND_REPO_TAG}" \
    -DTRITON_ENABLE_GPU="${TRITON_ENABLE_GPU}" \
    -DTRITON_ENABLE_STATS="${TRITON_ENABLE_STATS}" \
    -DRAPIDS_DEPENDENCIES_VERSION="${RAPIDS_DEPENDENCIES_VERSON}" \
    -DTRITON_FIL_USE_TREELITE_STATIC="${TRITON_FIL_USE_TREELITE_STATIC}" \
    -DTRITON_FIL_ENABLE_TREESHAP="${TRITON_FIL_ENABLE_TREESHAP}" \
    -DCMAKE_INSTALL_PREFIX=/rapids_triton/install \
    ..;

ENV CCACHE_DIR=/ccache

RUN --mount=type=cache,target=/ccache/ ninja install

# Stage for installing test dependencies
FROM base as base-test-install

COPY ./conda/environments/triton_test_no_client.yml /environment.yml

RUN --mount=type=cache,target=/root/miniconda3/pkgs conda env update -f /environment.yml \
    && rm /environment.yml

FROM base-test-install as wheel-install-0
RUN conda run --no-capture-output -n triton_test pip install tritonclient[all]

FROM ${SDK_IMAGE} as sdk-image

FROM base-test-install as wheel-install-1
COPY --from=sdk-image /workspace/install/python /sdk_install
RUN conda run --no-capture-output -n triton_test \
    pip install /sdk_install/tritonclient*manylinux*.whl \
 && rm -r /sdk_install

FROM wheel-install-${USE_CLIENT_WHEEL} as test-install
RUN conda run --no-capture-output -n triton_test \
    pip install git+https://github.com/rapidsai/rapids-triton.git@branch-21.12#subdirectory=python


# Stage for generating testing image
FROM ${SERVER_IMAGE} as test-stage

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=true
RUN if ! command -v conda; then \
      wget \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh \
      && mkdir /root/.conda \
      && bash Miniconda3-latest-Linux-$(uname -m).sh -b \
      && rm -f Miniconda3-latest-Linux-$(uname -m).sh \
      && conda init; \
    fi
ENV PYTHONDONTWRITEBYTECODE=false

COPY --from=test-install /root/miniconda3 /root/miniconda3

COPY qa /qa
COPY scripts /scripts

# Default to triton_test conda env
RUN sed -i s/rapids_triton_dev/triton_test/g /root/.bashrc

ENTRYPOINT []
CMD ["conda", "run", "--no-capture-output", "-n", "triton_test", "/qa/entrypoint.sh"]

FROM ${BASE_IMAGE} as final

ARG BACKEND_NAME
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
