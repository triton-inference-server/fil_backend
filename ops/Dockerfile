ARG TRITON_VERSION=21.10
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

FROM ${BASE_IMAGE} as base

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN [ $(uname -m) = 'x86_64' ] \
 && curl -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
 || curl -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-keyring_1.0-1_all.deb; \
 dpkg -i /tmp/cuda-keyring.deb \
 && rm /tmp/cuda-keyring.deb

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=true

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

COPY ./conda/environments/rapids_triton_dev_cuda11.4.yml /environment.yml

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete \
    && rm /root/miniconda3/envs/rapids_triton_dev/lib/libstdc++.so*

ENV PYTHONDONTWRITEBYTECODE=false

RUN mkdir /rapids_triton

COPY ./src /rapids_triton/src
COPY ./CMakeLists.txt /rapids_triton
COPY ./cmake /rapids_triton/cmake

WORKDIR /rapids_triton

SHELL ["conda", "run", "--no-capture-output", "-n", "rapids_triton_dev", "bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rapids_triton_dev", "bash"]

FROM base as build-stage

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BACKEND_NAME
ENV BACKEND_NAME=$BACKEND_NAME
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

RUN mkdir /rapids_triton/build /rapids_triton/install

WORKDIR /rapids_triton/build

# Remove potentially stale build artifacts
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    else \
      mkdir /opt/tritonserver/backends/${BACKEND_NAME}; \
    fi

RUN cmake \
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

RUN ninja install

FROM ${BASE_IMAGE}

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
