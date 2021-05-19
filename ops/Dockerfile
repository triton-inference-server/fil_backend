ARG TRITON_VERSION=21.04
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

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

ENV PYTHONDONTWRITEBYTECODE=false

COPY ./ /triton_fil_backend

WORKDIR /triton_fil_backend

FROM base as build-stage

ENV FIL_LIB=/opt/tritonserver/backends/fil/libtriton_fil.so
ENV LIB_DIR=/opt/lib/fil

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BUILD_TYPE=Release
ENV BUILD_TYPE=$BUILD_TYPE

# TODO: I am not sure why the lapack dependency is not picked up by ldd
RUN conda run --no-capture-output -n triton_dev \
   /triton_fil_backend/ops/build.sh \
 && cp -r /triton_fil_backend/build/install/backends/fil \
   /opt/tritonserver/backends/fil \
 && patchelf --set-rpath /root/miniconda3/envs/triton_dev/lib "$FIL_LIB" \
 && conda run --no-capture-output -n triton_dev \
   /bin/bash /triton_fil_backend/ops/move_deps.sh \
 && cp /root/miniconda3/envs/triton_dev/lib/liblapack.so.3 /opt/lib/fil \
 && patchelf --set-rpath /opt/lib/fil "$FIL_LIB"

RUN if [[ $BUILD_TYPE == 'Debug' ]]; \
    then \
      apt-get update \
      && apt-get install -y gdb valgrind \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*; \
    fi


FROM ${BASE_IMAGE}

COPY --from=build-stage /opt/lib/fil /opt/lib/fil

COPY --from=build-stage \
  /opt/tritonserver/backends/fil \
  /opt/tritonserver/backends/fil
