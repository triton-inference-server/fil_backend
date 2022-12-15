<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Building the FIL Backend
Triton backends are implemented as shared libraries which are conditionally
loaded by the main Triton server process. To build the FIL backend shared
library or simply to create a Docker image with a fresh build of the backend,
you may follow the indicated steps.

**Note**: Most users will not need to build their own copy of the FIL backend.
These instructions are intended for developers and those who wish to make
custom tweaks to the backend. If you are just looking for install instructions,
follow our [installation guide](docs/install.md).

## Prerequisites
The FIL backend may be built either using Docker or on the host. We
recommend using the Dockerized build in order to simplify dependency management
unless you have a specific need to build on the host.

### Dockerized Build
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Host Build
- [CUDA toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) (Only required for GPU-enabled builds)
- [CMake](https://cmake.org/install/)
- [Ninja](https://ninja-build.org/) (Optional but recommended)
Except for the CUDA toolkit, these dependencies can be installed via conda using the provided
[environment
file](https://github.com/triton-inference-server/fil_backend/blob/main/conda/environments/rapids_triton_dev.yml):

```bash
conda env create -f conda/environments/rapids_triton_dev.yml
conda activate rapids_triton_dev
```


## Using the Build Script
To simplify the build process, the FIL backend provides a `build.sh` script at
the root of the repo. For most use cases, it is sufficient to simply
invoke the script:

```bash
./build.sh
```

This is a lightweight wrapper around a `docker build` command which helps
provide the correct build arguments and variables. By default, it will build
*both* a "server" image which is equivalent to the usual Triton Docker image
and a "test" image whose entrypoint will invoke the FIL backend's tests.

### Build Options
The build script uses a number of flags and environment variables to
control the details of what gets built and how. These options are
summarized below:

#### Flags
- `-g`: Perform a debug build
- `-h`: Print help test for build script
- `--cpu-only`: Build CPU-only version of library
- `--no-treeshap`: Build without [treeshap
  support](https://github.com/triton-inference-server/fil_backend/blob/main/conda/environments/rapids_triton_dev.yml)
- `--tag-commit`: Tag Docker images using the current git commit
- `--no-cache`: Disable Docker cache for this build
- `--host`: Build on host, **not** in Docker
- `--buildpy`: Invoke Triton's `build.py` script to perform build.
  **Note:** This is **not** recommended for end-users. It is included
  primarily for testing compatibility with upstream build changes. If you must
  invoke this option, you will need the dependencies indicated in the
  associated conda [environment file](https://github.com/triton-inference-server/fil_backend/blob/main/conda/environments/buildpy.yml).

#### Environment variables
##### Standard options
- `BASE_IMAGE`: The base image for Docker images or the build image for
  `build.py` if `--buildpy` is invoked
- `TRITON_VERSION`: The version of Triton to use for this build
- `SERVER_TAG`: The tag to use for the server image
- `TEST_TAG`: The tag to use for the test image
- `PREBUILT_IMAGE`: An existing Triton Docker image which you would like to
  run tests against. This will build the test image on top of the indicated
  image.
- `RAPIDS_VERSION`: The version of RAPIDS to require for RAPIDS
  dependencies
##### Advanced options
- `USE_CLIENT_WHEEL`: If 1, the Triton Python client will be
  installed from a wheel distributed in the Triton SDK Docker image. This
  option is useful for ARM development, since the Triton client cannot
  currently be installed via `pip` for ARM.
- `SDK_IMAGE`: If set, this image will be used to provide the
  Python client wheel. Otherwise, if `USE_CLIENT_WHEEL` is set to 1 and this
  variable is unset, the image will be selected based on the Triton
  version.
- `CONDA_DEV_TAG`: A Docker image containing the development conda
  environment. Used primarily to speed up CI; rarely invoked during
  development.
- `CONDA_TEST_TAG`: A Docker image containing the test conda
  environment. Used primarily to speed up CI; rarely invoked during development
- `TRITON_REF`: The commit ref for the Triton server repo when using
  `--buildpy`
- `CORE_REF`: The commit ref for the Triton core repo when using
  `--buildpy`
- `COMMON_REF`: The commit ref for the Triton common repo when using
  `--buildpy`
- `BACKEND_REF`: The commit ref for the Triton backend repo when using
  `--buildpy`
- `THIRDPARTY_REF`: The commit ref for the Triton third-party repo when using
  `--buildpy`
- `JOB_ID`: Used for CI builds to uniquely identify a particular
  build job.
- `BUILDPY_BRANCH`: Use this branch of the Triton server repo to
  provide the `build.py` script if `--buildpy` is used.
- `TREELITE_STATIC`: if set to `ON`, Treelite will be statically linked into the built binaries
