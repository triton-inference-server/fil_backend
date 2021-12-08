###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Version of Triton to use
ARG TRITON_VERSION=21.11
# Base container image
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
# Whether or not to enable GPU build
ARG TRITON_ENABLE_GPU=ON
# A Triton server image to use as base for test layers (skip actual build)
ARG SERVER_IMAGE=build-stage

FROM ${BASE_IMAGE} as base

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget \
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
    -DBUILD_CUML_TESTS=OFF \
    -DBUILD_PRIMS_TESTS=OFF \
    -DBUILD_CUML_MG_TESTS=OFF \
    -DBUILD_CUML_EXAMPLES=OFF \
    -DBUILD_CUML_BENCH=OFF \
    -DBUILD_CUML_PRIMS_BENCH=OFF \
    -DCUML_CPP_ALGORITHMS=FIL \
    -DCMAKE_INSTALL_PREFIX=/rapids_triton/install \
    ..;

ENV CCACHE_DIR=/ccache

RUN --mount=type=cache,target=/ccache/ ninja install

# Stage for installing test dependencies
FROM base as test-install

COPY ./qa/environment.yml /environment.yml

RUN --mount=type=cache,target=/root/miniconda3/pkgs conda env update -f /environment.yml \
    && rm /environment.yml

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
