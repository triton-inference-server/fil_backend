###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Image to use if building from "vanilla" Ubuntu 20.04-based base container
ARG BASE_IMAGE=ubuntu:20.04
# Either the name of a cuDNN image or "cuda_base" to build from vanilla base
# container
ARG CUDA_IMAGE=cuda_base

# Whether or not to build indicated components
ARG FIL=1

###########################################################################################
# Base stage for all other Triton-based build and runtime stages
###########################################################################################
FROM ${BASE_IMAGE} as cuda_base

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/tritonserver/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Env variables users may expect from typical CUDA containers
ENV CUDA_VERSION 11.2.1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"
ENV CUDNN_VERSION 8.2.0.53

# TODO:
#    BOTH:
#        * cuda-libraries-11-2
#    RUNTIME:
#    BUILD:
#        * cuda-compiler-11-2
#        * cuda-libraries-dev-11-2
#        * cuda-nvml-dev-11-2

########################
# Install CUDA and cuDNN
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    | apt-key add - \
 && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" \
    > /etc/apt/sources.list.d/cuda.list \
 && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" \
    > /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    cuda-cudart-11-2=11.2.146-1 \
    cuda-compat-11-2 \
 && apt-get install -y \
    cuda-libraries-11-2 \
 && ln -s cuda-11.2 /usr/local/cuda \
 && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.3 \
 && apt-mark hold libcudnn8 \
 && rm -rf /var/lib/apt/lists/*

FROM ${CUDA_IMAGE} AS base
##################
# Install TensorRT

# NOTE: This file must be downloaded to the ops/stage directory by following
# the directions at
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
COPY ops/stage/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.0.3-ea-20210423_1-1_amd64.deb /tmp/tensorrt.deb

RUN dpkg -i /tmp/tensorrt.deb \
 && apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.0.3-ea-20210423/7fa2af80.pub \
 && apt-get update \
 && apt-get install -y \
    docker.io \
    libb64-dev \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvparsers-dev \
    libnvonnxparsers-dev \
    libnvinfer-samples \
    libre2-dev \
    libtool \
    onnx-graphsurgeon \
    pkg-config \
    python3-dev \
    python3-libnvinfer-dev \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    tensorrt \
 && rm /tmp/tensorrt.deb \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker && \
    pip3 install grpcio-tools grpcio-channelz

###########################################################################################
# Stage containing all Triton build dependencies
###########################################################################################
FROM base as build

RUN apt-get update \
 && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    cuda-compiler-11-2 \
    cuda-libraries-dev-11-2 \
    cuda-nvml-dev-11-2 \
    git \
    libopencv-dev \
    libssl-dev \
    libboost-dev \
    libcurl4-openssl-dev \
    patchelf \
    rapidjson-dev \
    unzip \
    uuid-dev \
    wget \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - \
    | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    cmake-data=3.18.4-0kitware1ubuntu20.04.1 \
    cmake=3.18.4-0kitware1ubuntu20.04.1 \
 && rm -rf /var/lib/apt/lists/*

# Retrieve Triton server source
ARG TRITON_VERSION=2.9.0
RUN mkdir /src \
 && cd /src/ \
 && wget https://github.com/triton-inference-server/server/archive/refs/tags/v${TRITON_VERSION}.tar.gz \
 && tar -xzf v${TRITON_VERSION}.tar.gz \
 && rm v${TRITON_VERSION}.tar.gz \
 && mkdir /src/server-${TRITON_VERSION}/build/output

# Build base Triton server
ARG PARALLEL=4
WORKDIR /src/server-${TRITON_VERSION}/build/output
RUN cmake .. \
 && make -j${PARALLEL} server \
 && mkdir -p /opt/tritonserver/backends \
 && mkdir /opt/lib \
 && cp -r server/install/* /opt/tritonserver

###########################################################################################
# Stage containing FIL backend build environment
###########################################################################################
FROM build as fil-base

ENV PATH="/root/miniconda3/bin:${PATH}"

ENV PYTHONDONTWRITEBYTECODE=true

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

COPY ./ops/environment.yml /environment.yml

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

ENV PYTHONDONTWRITEBYTECODE=false

COPY ./ /triton_fil_backend

WORKDIR /triton_fil_backend

###########################################################################################
# Stage in which FIL backend is built
###########################################################################################
FROM build as fil-build-0
FROM fil-base as fil-build-1

ENV FIL_LIB=/opt/tritonserver/backends/fil/libtriton_fil.so
ENV LIB_DIR=/opt/lib/fil

# TODO: I am not sure why the lapack dependency is not picked up by ldd
RUN conda run --no-capture-output -n triton_dev \
    /bin/bash /triton_fil_backend/ops/build.sh \
  && cp -r /triton_fil_backend/build/install/backends/fil \
    /opt/tritonserver/backends/fil \
  && patchelf --set-rpath /root/miniconda3/envs/triton_dev/lib "$FIL_LIB" \
  && conda run --no-capture-output -n triton_dev \
    /bin/bash /triton_fil_backend/ops/move_deps.sh \
  && cp /root/miniconda3/envs/triton_dev/lib/liblapack.so.3 /opt/lib/fil \
  && patchelf --set-rpath /opt/lib/fil "$FIL_LIB"

FROM fil-build-${FIL} as fil-build

###########################################################################################
# Stage for staging all built Triton features
###########################################################################################
FROM build as final-build

COPY --from=fil-build /opt/lib/fi[l] /opt/lib/fil

COPY --from=fil-build \
  /opt/tritonserver/backends/fi[l] \
  /opt/tritonserver/backends/fil

###########################################################################################
# Stage containing only Triton runtime dependencies
###########################################################################################
FROM base as runtime
COPY --from=final-build /opt/lib /opt/lib
COPY --from=final-build /opt/tritonserver /opt/tritonserver
