FROM nvcr.io/nvidia/tritonserver:21.02-py3 as base

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# TODO: REMOVE GDB
RUN apt-get update \
    && apt-get install --no-install-recommends -y wget \
    gdb \
    valgrind \
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

FROM base as dev-stage

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "triton_dev", "/bin/bash"]

FROM base as build-stage


RUN conda run --no-capture-output -n triton_dev \
  /bin/bash /triton_fil_backend/ops/build.sh \
  && cp -r /triton_fil_backend/build/install/backends/fil \
    /opt/tritonserver/backends/fil

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/miniconda3/envs/triton_dev/lib"

WORKDIR /opt/tritonserver

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "triton_dev", "/opt/tritonserver/nvidia_entrypoint.sh"]

# FROM nvcr.io/nvidia/tritonserver:21.02-py3
# 
# COPY --from=build-stage \
#   /triton_fil_backend/build/install/backends/fil \
#   /opt/tritonserver/backends/fil
# 
# COPY --from=build-stage /root/miniconda3/envs/triton_dev/lib/libtreelite.so /lib
