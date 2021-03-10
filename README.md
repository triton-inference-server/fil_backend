<!--
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Inference Server Identity Backend


```
sudo apt-get install -y rapidjson-dev
sudo apt-get install -y libnccl2 libnccl-dev
sudo apt-get install -y libblas-dev liblapack-dev
sudo apt-get install -y python3.8
sudo apt-get install -y python-is-python3
apt-get install python-dev python3-dev
sudo apt-get install intel-mkl-full
pip3 install numpy
pip3 install treelite
git clone https://github.com/dmlc/treelite.git
cd treelite && mkdir build && cd build
cmake .. && make && make install
sudo apt-get install -y libgtest-dev
sudo apt-get install -y doxygen


FAISS
cmake -B build .
make
make install

CUML
cmake .. \
  -DSINGLEGPU=ON \
  -DBUILD_STATIC_FAISS=ON \
  -DBUILD_CUML_TESTS=OFF \
  -DBUILD_PRIMS_TESTS=OFF \
  -DBUILD_CUML_EXAMPLES=OFF \
  -DBUILD_CUML_BENCH=OFF \
  -DBUILD_CUML_PRIMS_BENCH=OFF \
  -DGPU_ARCHS="70"

cmake .. \
  -DBUILD_STATIC_FAISS=ON \
  -DBUILD_RAFT_TESTS=OFF


libcuml++

```

An example Triton backend that demonstrates most of the Triton Backend
API. You can learn more about backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues).

Use cmake to build and install in a local directory.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]
