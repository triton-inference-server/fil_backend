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

# Installation
The FIL backend is a part of Triton and can be installed via the methods
described in the [main Triton
documentation](https://github.com/triton-inference-server/server#build-and-deploy).
To quickly get up and running with a Triton Docker image, follow these
steps.

**Note**: Looking for instructions to *build* the FIL backend yourself? Check out our [build
guide](https://github.com/triton-inference-server/fil_backend/blob/main/docs/build.md).

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Getting the container
Triton containers are available from NGC and may be pulled down via

```bash
docker pull nvcr.io/nvidia/tritonserver:22.10-py3
```

Note that the FIL backend cannot be used in the `21.06` version of this
container; the `21.06.1` patch release is the earliest Triton version with a
working FIL backend implementation.

## Starting the container
In order to actually deploy a model, you will need to provide the serialized
model and configuration file in a specially-structured directory called the
"model repository." Check out the
[configuration guide](https://github.com/triton-inference-server/fil_backend/blob/main/docs/configuration.md) for details on how to do this for your model.

Assuming your model repository is  on your host system, you can
bind-mount it into the container and start the server via the following
command:
```
!docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v {MODEL_REPO}:/models --name tritonserver
```
