FROM ubuntu:20.04 as cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.2.1

# TODO: Install components of toolkit more selectively
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-2=11.2.146-1 \
    cuda-compat-11-2 \
    && apt-get install -y cuda-toolkit-11-2 \
    && ln -s cuda-11.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"

FROM cuda as cudnn

ENV CUDNN_VERSION 8.2.0.53

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.3 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

FROM cudnn as tensorrt
ADD ops/stage/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.0.3-ea-20210423_1-1_amd64.deb /tmp/tensorrt.deb

RUN dpkg -i /tmp/tensorrt.deb \
 && apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.0.3-ea-20210423/7fa2af80.pub \
 && apt-get update \
 && apt-get install -y \
   libnvinfer-dev \
   libnvinfer-plugin-dev \
   libnvparsers-dev \
   libnvonnxparsers-dev \
   libnvinfer-samples \
   tensorrt \
   python3-libnvinfer-dev \
   onnx-graphsurgeon \
 && rm /tmp/tensorrt.deb

FROM tensorrt as triton

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            docker.io \
            git \
            libopencv-dev \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            libcurl4-openssl-dev \
            libb64-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            python3-opencv \
            rapidjson-dev \
            software-properties-common \
            unzip \
            wget \
            zlib1g-dev \
            pkg-config \
            uuid-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker && \
    pip3 install grpcio-tools grpcio-channelz

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake-data=3.18.4-0kitware1ubuntu20.04.1 cmake=3.18.4-0kitware1ubuntu20.04.1

RUN mkdir /src

WORKDIR /src

RUN wget https://github.com/triton-inference-server/server/archive/refs/tags/v2.9.0.tar.gz \
 && cd /src/ \
 && tar -xzf v2.9.0.tar.gz \
 && mkdir /src/server-2.9.0/build/output \
 && cd /src/server-2.9.0/build/output \
 && cmake .. \
 && make -j12 server \
 && mkdir /opt/tritonserver \
 && cp -r server/install/* /opt/tritonserver

ENV PATH="/opt/tritonserver/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib:${LD_LIBRARY_PATH}"

FROM triton as fil-base

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

FROM fil-base as fil-build

ENV FIL_LIB=/opt/tritonserver/backends/fil/libtriton_fil.so
ENV LIB_DIR=/opt/lib/fil

# TODO: I am not sure why the lapack dependency is not picked up by ldd
# TODO: Move backends folder creation to Triton base
RUN conda run --no-capture-output -n triton_dev \
    /bin/bash /triton_fil_backend/ops/build.sh \
  && mkdir /opt/tritonserver/backends \
  && cp -r /triton_fil_backend/build/install/backends/fil \
    /opt/tritonserver/backends/fil \
  && patchelf --set-rpath /root/miniconda3/envs/triton_dev/lib "$FIL_LIB" \
  && conda run --no-capture-output -n triton_dev \
    /bin/bash /triton_fil_backend/ops/move_deps.sh \
  && cp /root/miniconda3/envs/triton_dev/lib/liblapack.so.3 /opt/lib/fil \
  && patchelf --set-rpath /opt/lib/fil "$FIL_LIB"


FROM triton

COPY --from=fil-build /opt/lib/fil /opt/lib/fil

COPY --from=fil-build \
  /opt/tritonserver/backends/fil \
  /opt/tritonserver/backends/fil
