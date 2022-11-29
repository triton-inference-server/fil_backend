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

# Repo Overview

The FIL backend repo is organized in the following directories:

## `ci`
This directory contains scripts and configuration files for working with CI.
Developers may invoke `ci/local/build.sh` to build and run tests locally or
`ci/gitlab/build.sh` to more precisely mirror the test environment run in
official CI. This directory is not intended for end-users.

## `cmake`
This directory contains CMake files required for the build, especially those
which are used to retrieve external dependencies. It is not intended for
end-users

## `conda`
This directory contains conda-related infrastructure including environment yaml
files used to construct build and test environments:

- `conda/environments/buildpy.yml`: Minimal environment for using Triton's
  `build.py` build script
- `conda/environments/rapids_triton_dev.yml`: Environment for building the FIL
  backend
- `conda/environments/triton_benchmark.yml`: Environment for running the FIL
  backend's standard benchmarks
- `conda/environments/triton_test_no_client.yml`: Environment for running tests
  for the FIL backend. This file does not include Triton's Python client to
  facilitate testing on ARM machines, where the client cannot be correctly
  installed via pip.
- `conda/environments/triton_test.yml`: Environment for running tests for the
  FIL backend that includes Triton's Python client. Recommended environment for
  those wishing to run tests outside of Docker.

## `docs`
This directory contains markdown files for documentation.

## `notebooks`
This directory contains example Jupyter notebooks for using the FIL backend.

## `ops`
This directory contains files used for build-related tasks including the
Dockerfile for the FIL backend's dockerized build. It is not intended for end
users.

## `qa`
This directory contains files for running tests and benchmarks. It is not
intended for end-users.

## `scripts`
This directory contains utility scripts for e.g. converting models to Treelite
checkpoint format. It also contains a conda environment file indicating the
necessary dependencies for running these scripts.

## `src`
This directory contains the C++ source files for the FIL backend. It is not
intended for end-users.
