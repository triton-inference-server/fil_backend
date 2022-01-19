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

# Fraud Detection With Categorical XGBoost

This example notebook shows how to train and deploy an XGBoost model
with categorical features in Triton using the FIL backend. We begin by training
two separate models on a fraud detection dataset with categorical variables:
one small model designed to maximize runtime performance and one larger model
designed to maximize accurate and precise detection of fraud. We then deploy
both models on CPU and GPU and compare their performance using Triton's
`perf_analyzer`. Based on these results, we see that GPU deployment opens up
the possibility of deploying a much larger and more accurate fraud model with
higher throughput while also keeping to a tight latency budget.

## Running the notebook
The notebook itself requires a few Python dependencies. To easily install them,
you may make use of the provided conda [environment
file](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection/environment.yml)
as follows:
```bash
conda env create -f environment.yml
```
You may then activate the conda environment and run the notebook as usual:
```bash
conda activate triton_example
jupyter notebook
```
The Jupyter interface should now be accessible from a browser, and you can
follow the instructions from there.

**NOTE**: You will also need [Docker](https://docs.docker.com/get-docker/)
available on your system in order to run the Triton server as part of the
example notebook.
