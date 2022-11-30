<!--
# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Triton Inference Server FIL Backend

Triton is a machine learning inference server for easy and highly optimized
deployment of models trained in almost any major framework. This backend
specifically facilitates use of tree models in Triton (including models trained
with [XGBoost](https://xgboost.readthedocs.io/en/stable/),
[LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/),
[Scikit-Learn](https://scikit-learn.org/stable/), and
[cuML](https://docs.rapids.ai/api/cuml/stable/)).

**If you want to deploy a tree-based model for optimized real-time or
batched inference in production, the FIL backend for Triton will allow you to
do just that.**

## Table of Contents
### Usage Information
- [Installation](https://github.com/triton-inference-server/fil_backend/blob/main/docs/install.md)
- [Introductory end-to-end
  example](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/categorical-fraud-detection/Fraud_Detection_Example.ipynb)
- [FAQ
  notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb) with code snippets for many common scenarios
- [Model Configuration](https://github.com/triton-inference-server/fil_backend/blob/main/docs/model_config.md)
- [Explainability and Shapley value support](https://github.com/triton-inference-server/fil_backend/blob/main/docs/explainability.md)
- [Scikit-Learn and cuML model support](https://github.com/triton-inference-server/fil_backend/blob/main/docs/sklearn_and_cuml.md)
- [Model support and limitations](https://github.com/triton-inference-server/fil_backend/blob/main/docs/model_support.md)

### Contributor Docs
- [Development workflow](https://github.com/triton-inference-server/fil_backend/blob/main/docs/workflow.md)
- [Overview of the repo](https://github.com/triton-inference-server/fil_backend/blob/main/docs/repo_overview.md)
- [Build instructions](https://github.com/triton-inference-server/fil_backend/blob/main/docs/build.md)
- [Running tests](https://github.com/triton-inference-server/fil_backend/blob/main/docs/tests.md)
- [Making a contribution](https://github.com/triton-inference-server/fil_backend/blob/main/CONTRIBUTING.md)

## Not sure where to start?
If you aren't sure where to start with this documentation, consider one of the
following paths:

- **I currently use XGBoost/LightGBM or other tree models and am trying to
   assess if Triton is the right solution for production deployment of my
   models**
   1. Check out the FIL backend's [blog post announcement](https://developer.nvidia.com/blog/real-time-serving-for-xgboost-scikit-learn-randomforest-lightgbm-and-more/)
   2. Make sure your model is supported by looking at the [model support](https://github.com/triton-inference-server/fil_backend/blob/main/docs/model_support.md) section
   2. Look over the [introductory example](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/categorical-fraud-detection/Fraud_Detection_Example.ipynb)
   3. Try deploying your own model locally by consulting the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb).
   4. Check out the main [Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server) for additional features and helpful tips on deployment (including example [Helm charts](https://github.com/triton-inference-server/server/blob/main/deploy/gcp/README.md#kubernetes-deploy-triton-inference-server-cluster)).
- **I am familiar with Triton, but I am using it to deploy an XGBoost/LightGBM
model for the first time.**
   1. Look over the [introductory example](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/categorical-fraud-detection/Fraud_Detection_Example.ipynb)
   2. Try deploying your own model locally by consulting the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb). Note that it includes specific example code for serialization of [XGBoost](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#Example-1.1:-Serializing-an-XGBoost-model) and [LightGBM](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#Example-1.2-Serializing-a-LightGBM-model) models.
   3. Review the FAQ notebook's [tips](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#Example-9:-Optimizing-model-performance) for optimizing model performance.
- **I am familiar with Triton and the FIL backend, but I am using it to deploy a Scikit-Learn or cuML tree model for the first time**
    1. Look at the section on preparing [Scikit-Learn/cuML models](https://github.com/triton-inference-server/fil_backend/blob/main/docs/sklearn_and_cuml.md) for Triton.
    2. Try deploying your model by consulting the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb), especially the sections on [Scikit-Learn and cuML](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1.3-Can-I-deploy-Scikit-Learn/cuML-models-serialized-with-Pickle?).
- **I am a data scientist familiar with tree model training, and I am trying
   to understand how Triton might be used with my models.**
   1. Take a glance at the [Triton product page](https://developer.nvidia.com/nvidia-triton-inference-server) to get a sense of what Triton is used for.
   2. Download and run the [introductory example](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection) for yourself. If you do not have access to a GPU locally, you can just look over this notebook and then jump to the [FAQ notebook](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/faq) which has specific information on CPU-only training and deployment.
4. **I have never worked with tree models before.**
   1. Take a look at XGBoost's [documentation](https://xgboost.readthedocs.io/en/stable/get_started.html#python).
   2. Download and run the [introductory example](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection) for yourself.
   3. Try deploying your own model locally by consulting the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb).
- **I don't like reading docs.**
    1. Look at the
       [Quickstart](https://github.com/triton-inference-server/fil_backend#quickstart-deploying-a-tree-model-in-3-steps) below
    2. Open the [FAQs notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb) in a browser.
    3. Try deploying your model. If you get stuck, `Ctrl-F` for keywords on the
       FAQ page.


## Quickstart: Deploying a tree model in 3 steps
1. Copy your model into the following directory structure. In this
   example, we show an XGBoost json file, but XGBoost binary files,
   LightGBM text files, and Treelite checkpoint files are also supported.
```
model_repository/
├─ example/
│  ├─ 1/
│  │  ├─ model.json
│  ├─ config.pbtxt
```
2. Fill out config.pbtxt as follows, replacing `$NUM_FEATURES` with the number
   of input features, `$MODEL_TYPE` with `xgboost`, `xgboost_json`,
   `lightgbm` or `treelite_checkpoint`, and `$IS_A_CLASSIFIER` with `true`
   or `false` depending on whether this is a classifier or regressor.
```
backend: "fil"
max_batch_size: 32768
input [
 {  
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ $NUM_FEATURES ]                    
  } 
]
output [
 {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
instance_group [{ kind: KIND_AUTO }]
parameters [
  {
    key: "model_type"
    value: { string_value: "$MODEL_TYPE" }
  },
  {
    key: "output_class"
    value: { string_value: "$IS_A_CLASSIFIER" }
  }
]

dynamic_batching {}
```
3. Start the server:
```
docker run -p 8000:8000 -p 8001:8001 --gpus all \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:22.11-py3 \
  tritonserver --model-repository=/models
```

The Triton server will now be serving your model over both HTTP (port 8000)
and GRPC (port 8001) using NVIDIA GPUs if they are available or the CPU if
they are not. For information on how to [submit inference
requests](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-5:-How-do-I-submit-an-inference-request-to-Triton?), how to
deploy [other tree model types](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1:-What-can-I-deploy-with-the-FIL-backend?), or [advanced configuration options](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-9:-How-can-we-improve-performance-of-models-deployed-with-the-FIL-backend?), check out the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb).
