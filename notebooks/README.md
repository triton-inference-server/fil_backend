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

# FIL Backend Examples

This directory contains example notebooks which illustrate typical workflows
and use-cases for the Triton FIL backend. Additional examples will be added to
this directory over time.

**Note**: For those looking for the example notebook which was initially
included at the top-level of this directory, it has been moved to the
[simple-xgboost](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/simple-xgboost)
directory.

Each subdirectory contains an example notebook and a README with instructions
on how to run the example.

## Current Examples
- [Simple
  XGBoost](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/simple-xgboost):
  This example demonstrates how to load an XGBoost model into Triton and
  optimize its configuration with the Triton Model Analyzer.
- [Categorical Fraud
  Example](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection):
  This example walks through training a categorical XGBoost model for fraud
  detection and deploying it on both GPU-accelerated and CPU-only systems.
