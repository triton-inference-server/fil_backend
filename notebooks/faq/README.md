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

# FAQs and Advanced Features

Designed as a complete reference to features of the FIL backend and common
tasks performed with it, this notebook provides answers to a series of FAQs
along with code snippets demonstrating how to make practical use of those
answers.

If you have never made use of the FIL backend before, it is recommended that
you begin with the introductory [fraud detection notebook](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection#fraud-detection-with-categorical-xgboost). After working through this basic example, the FAQs notebook will offer answers to questions that go beyond the basics in order to get the most out of the FIL backend.

## Running the notebook
In order to launch the Triton server, you will need
[Docker](https://docs.docker.com/get-docker/) installed on your system. The
rest of the notebook also requires a few Python dependencies. To easily install
these additional dependencies, you may make use of the provided conda
[environment
file](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/faq/environment.yml)
as follows:
```bash
conda env create -f environment.yml
```
You may then activate the conda environment and run the notebook as usual:
```bash
conda activate triton_faq_nb
jupyter notebook
```
The Jupyter interface should now be accessible from a browser, and you can
follow the instructions within the notebook itself from there.

Note that depending on which model framework you choose to use with this
notebook, you may not need all the dependencies listed in the conda environment
file. Remove any that you do not wish to install before installing the
environment.
