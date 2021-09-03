ARG TRITON_VERSION=21.07
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

FROM ${BASE_IMAGE} as base

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

RUN cat /environment.yml \
    && conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete \
    && rm /root/miniconda3/envs/triton_dev/lib/libstdc++.so*

ENV PYTHONDONTWRITEBYTECODE=false

COPY ./ /triton_fil_backend

WORKDIR /triton_fil_backend

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "triton_dev", "bash"]

FROM base as build-stage

ENV FIL_LIB=/opt/tritonserver/backends/fil/libtriton_fil.so
ENV LIB_DIR=/opt/tritonserver/backends/fil/deps

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BUILD_TYPE=Release
ENV BUILD_TYPE=$BUILD_TYPE

RUN [ -d /opt/tritonserver/backends ] || mkdir -p /opt/tritonserver/backends

# Remove potentially stale dependencies
RUN if [ -d /opt/tritonserver/backends/fil ]; \
    then \
      rm -rf /opt/tritonserver/backends/fil/*; \
    fi

# TODO: I am not sure why the lapack dependency is not picked up by ldd
RUN conda run --no-capture-output -n triton_dev \
   /triton_fil_backend/ops/build.sh \
 && cp -r /triton_fil_backend/build/install/backends/fil \
   /opt/tritonserver/backends \
 && conda run --no-capture-output -n triton_dev \
   python3 /triton_fil_backend/ops/move_deps.py \
 && cp /root/miniconda3/envs/triton_dev/lib/liblapack.so.3 "$LIB_DIR"

RUN if [[ $BUILD_TYPE == 'Debug' ]]; \
    then \
      apt-get update \
      && apt-get install -y gdb valgrind \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*; \
    fi

FROM ${BASE_IMAGE}

# Remove existing FIL backend install
RUN if [ -d /opt/tritonserver/backends/fil ]; \
    then \
      rm -rf /opt/tritonserver/backends/fil/*; \
    fi

COPY --from=build-stage \
  /opt/tritonserver/backends/fil \
  /opt/tritonserver/backends/fil
