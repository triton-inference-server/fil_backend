<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# Customized End-to-End Builds

For a variety of reasons, you may wish to build the Triton server with FIL
backend from a different base container or with a greater degree of control
over the build process than the main Dockerfile provides. In such cases, you
can make use of
[ops/e2e.Dockerfile](https://github.com/wphicks/triton_fil_backend/blob/main/ops/e2e.Dockerfile)
and the
[ops/e2e_build.sh](https://github.com/wphicks/triton_fil_backend/blob/main/ops/e2e_build.sh)
wrapper script. This Dockerfile explicitly lays out every build step from a
vanilla Ubuntu 20.04 Docker image, allowing you to build on any Ubuntu
20.04-based image you like.

NOTE: In its current state, this Dockerfile builds only basic features of the
Triton server and includes only the FIL backend. Later work may add the option
to build additional features. Please [file a feature
request](https://github.com/wphicks/triton_fil_backend/issues) if there is
something you'd like to see added.

## Prerequisites
To perform an end-to-end build, you will need:

- [Docker >= 18.09](https://docs.docker.com/get-docker/)
- The TensorRT 8.0.0.3 install .deb file for Ubuntu 20.04

The TensorRT install file can be downloaded by following [these
instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading).
Note that you will need an NVIDIA Developer Program login to perform the
download, but it is free and easy to [sign
up](https://developer.nvidia.com/developer-program). You should download the
.deb to
`ops/stage/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.0.3-ea-20210423_1-1_amd64.deb`.

## Building with `e2e_build.sh`

The easiest way to perform an end-to-end build is to make use of the
`ops/e2e_build.sh` script. A few environment variables are provided to allow
you to customize the details of the build including:
- `BASE_IMAGE`: The Ubuntu 20.04-based base image to build from. (Default:
  [`ubuntu:20.04`](https://hub.docker.com/layers/ubuntu/library/ubuntu/20.04/images/sha256-1de4c5e2d8954bf5fa9855f8b4c9d3c3b97d1d380efe19f60f3e4107a66f5cae?context=explore)
- `TAG`: The tag to use for the built image (Default: `triton_fil`)
- `PARALLEL`: What concurrency level to use for the build. Defaults to the
  output of `nproc`
- `FIL`: 1 or 0. A value of 1 indicates that the FIL backend *should* be built
  as part of this build. If given a value of 0, the Triton server will still be
  built, but the FIL backend will not be included. (Default: 1)

### Example invocation
```bash
PARALLEL=4 BASE_IMAGE=ubuntu:20.04 TAG=triton_fil FIL=1 ./ops/build.sh
```
NOTE: This script should be invoked from the repository root.

## Building without `e2e_build.sh`
Most of the options for `e2e_build.sh` are simply Docker build args, so you may
perform a build by invoking Docker directly. Note that the default value for
`FIL` is reversed for the Dockerfile from the default used in `e2e_build.sh` in
order to maintain compatibility with other usages. Therefore, if you wish to
build the FIL backend, be sure to pass `FIL=1` as a build arg.

It is also *strongly* recommended that you [enable
BuildKit](https://docs.docker.com/develop/develop-images/build_enhancements/)
in order to avoid unnecessary steps in the build process, as shown in the
following invocation (from the repository root):

```bash
DOCKER_BUILDKIT=1 docker build \
  -t triton_fil \
  -f ops/e2e.Dockerfile \
  --build-arg PARALLEL=4 \
  --build-arg BASE_IMAGE=ubuntu:20.04 \
  --build-arg FIL=1 \
  .
```
NOTE: The Dockerfile contains a few build args not listed here. Those build
args should be considered experimental and untested but are made available in
case they are required for specific use cases.

## Usage
Once built, this Triton server image may be used exactly as the server image
described in the [main
README](https://github.com/wphicks/triton_fil_backend#running-the-server).
Remember, however, that this image does not include all of the other backends
and features built into the main image.
