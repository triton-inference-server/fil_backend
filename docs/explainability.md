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

# Model Explainability with Shapley Values

**NOTE: The CPU version of this feature is in an experimental state as of version 23.04**

In addition to providing model output from forest models, the FIL backend
can help you understand *why* the model came to a particular conclusion by
providing Shapley values. Shapley values offer a measure of the extent to
which individual features in an input contributed to the final model output.
Features with high Shapley value scores can generally be understood to be more
important to the model's conclusion than those with lower scores.

Generally speaking, Shapley values are computed by computing the model output
with and without a particular feature input and looking at how much the output
changed. This is referred to as the marginal contribution of that
feature. For a more complete understanding, check out the [Wikipedia
article](https://en.wikipedia.org/wiki/Shapley_value) on Shapley values or
Lloyd Shapley's [original
paper](https://www.rand.org/content/dam/rand/pubs/research_memoranda/2008/RM670.pdf).

**NOTE: Tree depth is limited to 32 for shapley value computation. Tree models with higher depth will throw an error.**

## Using Shapley Values in the FIL Backend
Because it takes additional time to compute and return the relatively large
output arrays for Shapley values, Shapley value computation is turned off by
default in the FIL backend.

To turn on Shapley Value support, you must add an additional output to the
`config.pbtxt` file for your model as shown below:
```protobuf
output [
 {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  },
 {
    name: "treeshap_output"
    data_type: TYPE_FP32
    dims: [ 501 ]
  }
]
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
 },
 {
    name: "treeshap_output"
    data_type: TYPE_FP32
    dims: [ $NUM_FEATURES_PLUS_ONE ]
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
Note that the length of the `treeshap_output` is equal to the number of input
features plus one to account for the bias term in the Shapley output. For a
working example of model deployment with Shapley values, including how to
retrieve those values using Triton's Python client, check out the [FAQ
Notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#$\color{#76b900}{\text{FAQ-12:-How-do-I-retrieve-Shapley-values-for-model-explainability?}}$)
