ARG BASE_IMAGE=triton_fil
FROM $BASE_IMAGE

ENV PATH="/root/miniconda3/bin:${PATH}"

COPY NONEXISTENT /NONEXISTENT/NONEXISTENT

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

# Use mamba to substantially improve speed of environment install
RUN conda install mamba -n base -c conda-forge

COPY ./environment.yml /environment.yml

RUN conda create -n triton_test \
    && mamba install -n triton_test python=3.8 pip \
    && conda run -n triton_test pip install nvidia-pyindex \
    && mamba env update -f /environment.yml \
    && rm /environment.yml \
    && mamba clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

ENV PYTHONDONTWRITEBYTECODE=false

RUN mkdir /logs

COPY ./run_tests.sh /entrypoint.sh
COPY ./L0_e2e /L0_e2e

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "triton_test"]
CMD ["bash", "/entrypoint.sh"]
