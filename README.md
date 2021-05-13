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

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Triton Inference Server FIL Backend

This backend allows forest models trained by several popular machine learning
frameworks to be deployed in a [Triton inference
server](https://developer.nvidia.com/nvidia-triton-inference-server) using
the RAPIDS [Forest Inference
LIbrary](https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35)
for fast GPU-based inference. Using this backend, forest models can be deployed
seamlessly alongside deep learning models for fast, unified inference
pipelines.

*WARNING:* This is preview-quality software. It is offered here as part of
ongoing collaborative efforts to determine future direction for this project
but should not be considered stable enough for production use until a later
release.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Installation

#### Pre-built container

Pre-built Triton containers containing the FIL backend should eventually be
available from NGC. In the meantime, you will need to build the container
manually.

#### Building locally

To build the Triton server container with the FIL backend, run the following
from the repo root:

```bash
docker build -t triton_fil -f ops/Dockerfile .
```

### Running the server

Before starting the server, you will need to set up a "model repository"
directory containing the model you wish to serve as well as a configuration
file. The FIL backend currently supports forest models serialized in XGBoost's
binary format, XGBoost's JSON format, and LightGBM's text format. See the
following example directory structure for an XGBoost binary file:

```
model_repository/
`-- fil
    |-- 1
    |   `-- xgboost.model
    `-- config.pbtxt
```

By default, the FIL backend assumes that XGBoost binary models will be named
`xgboost.model`, XGBoost json models will be named `xgboost.json`, and LightGBM
models will be named `model.txt`, but this can be tweaked through standard
Triton configuration options.

#### Configuration

Once you have chosen a model to deploy, you will need to create a corresponding
`config.pbtxt` file. An example of this configuration file is shown below:

```
name: "fil"
backend: "fil"
max_batch_size: 8192
input [
 {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 500 ]
  }
]
output [
 {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]
instance_group [{ kind: KIND_GPU }]
parameters [
  {
    key: "model_type"
    value: { string_value: "xgboost" }
  },
  {
    key: "predict_proba"
    value: { string_value: "true" }
  },
  {
    key: "output_class"
    value: { string_value: "true" }
  },
  {
    key: "threshold"
    value: { string_value: "0.5" }
  },
  {
    key: "algo"
    value: { string_value: "ALGO_AUTO" }
  },
  {
    key: "storage_type"
    value: { string_value: "AUTO" }
  },
  {
    key: "blocks_per_sm"
    value: { string_value: "0" }
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8, 16, 32, 64, 128, 1024, 8192]
  max_queue_delay_microseconds: 30000
}
```

For a full description of the configuration schema, see the Triton [server
docs](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).
Here, we will simply summarize the most commonly-used options and those
specific to FIL:

- `max_batch_size`: The maximum number of samples to process in a batch. In
  general, FIL's efficient handling of even large forest models means that this
  value can be quite high (2^13 in the example), but this may need to be
  reduced if you find that you are exhausting system resources. (NOTE: Due to a
  [current bug](https://github.com/wphicks/triton_fil_backend/issues/40), this
  value should not be set to 16384 or higher.
- `input`: This configuration block specifies information about the input
  arrays that will be provided to the FIL model. The `dims` field should be set
  to `[ NUMBER_OF_FEATURES ]`, but all other fields should be left as they
  are in the example.
- `output`: This configuration block specifies information about the arrays
  output by the FIL model. If the `predict_proba` option (described later) is
  set to "true" and you are using a classification model, the `dims` field
  should be set to `[ NUMBER_OF_CLASSES ]`. Otherwise, this can simply be `[ 1 ]`,
  indicating that the model returns a single class ID for each sample.

- `parameters`: This block contains FIL-specific configuration details. Note
  that all parameters are input as strings and should be formatted with `key`
  and `value` fields as shown in the example.
  * `model_type`: One of `"xgboost"`, `"xgboost_json"`, or `"lightgbm"`,
    indicating whether the provided model is in XGBoost binary format, XGBoost
    JSON format or LightGBM text format respectively.
  * `predict_proba`: Either `"true"` or `"false"`, depending on whether the
    desired output is a score for each class or merely the predicted class ID.
  * `output_class`: Either `"true"` or `"false"`, depending on whether the
    model is a classification or regression model.
  * `threshold`: The threshold score used for class prediction.
  * `algo`: One of `"ALGO_AUTO"`, `"NAIVE"`, `"TREE_REORG"` or
    `"BATCH_TREE_REORG"` indicating which FIL inference algorithm to use. More
    details are available in the [cuML
    documentation](https://docs.rapids.ai/api/cuml/stable/api.html?highlight=algo_t#cuml.ForestInference.load_from_treelite_model),
    but `"ALGO_AUTO"` is a safe choice for all models.
  * `storage_type`: One of `"AUTO"`, `"DENSE"`, `"SPARSE"`, and `"SPARSE8"`, indicating
    the storage format that should be used to represent the imported model.
    `"AUTO"` indicates that the storage format should be automatically chosen.
    `"SPARSE8"` is currently experimental.
  * `blocks_per_sm`: If set to any nonzero value (generally between 2 and 7),
    this provides a limit to improve the cache hit rate for large forest
    models. In general, network latency will significantly overshadow any
    speedup from tweaking this setting, but it is provided for cases where
    maximizing throughput is essential.
- `dynamic_batching`: This configuration block specifies how Triton should
  perform dynamic batching for your model. Full details about these options can
  be found in the main [Triton
  documentation](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#models-and-schedulers). You may find it useful to test your configuration using the [Triton `perf_analyzer` tool](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md) in order to optimize performance.
  * `preferred_batch_size`: A list of preferred values for the number of
    samples to process in a single batch.
  * `max_queue_delay_microseconds`: How long of a window in which requests can
    be accumulated to form a batch of a preferred size.

Note that the configuration is in protobuf format. If invalid protobuf is
provided, the model will fail to load, and you will see an error line in the
server log containing `Error parsing text-format inference.ModelConfig:`
followed by the line and column number where the parsing error occurred.

#### Starting the server
To run the server with the configured model, execute the following command:
```bash
docker run \
  --gpus=all \
  --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $PATH_TO_MODEL_REPO_DIR:/models \
  triton_fil \
  tritonserver \
  --model-repository=/models
```

#### Submitting inference requests
General examples for submitting inference requests to a Triton server are
available
[here](https://github.com/triton-inference-server/server/blob/master/docs/client_examples.md).
For convenience, we provide the following example code for using the Python
client to submit inference requests to a FIL model deployed on a Triton server
on the local machine:

```python
import numpy
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc

# Set up both HTTP and GRPC clients. Note that the GRPC client is generally
# somewhat faster.
http_client = triton_http.InferenceServerClient(
    url='localhost:8000',
    verbose=False,
    concurrency=12
)
grpc_client = triton_grpc.InferenceServerClient(
    url='localhost:8001',
    verbose = False
)

# Generate dummy data to classify
features = 1_000
samples = 8_192
data = numpy.random.rand(samples, features).astype('float32')

# Set up Triton input and output objects for both HTTP and GRPC
triton_input_http = triton_http.InferInput(
    'input__0',
    (samples, features),
    'FP32'
)
triton_input_http.set_data_from_numpy(data, binary_data=True)
triton_output_http = triton_http.InferRequestedOutput(
    'output__0',
    binary_data=True
)
triton_input_grpc = triton_grpc.InferInput(
    'input__0',
    (samples, features),
    'FP32'
)
triton_input_grpc.set_data_from_numpy(data)
triton_output_grpc = triton_grpc.InferRequestedOutput('output__0')

# Submit inference requests (both HTTP and GRPC)
request_http = http_client.infer(
    'fil',
    model_version='1',
    inputs=[triton_input_http],
    outputs=[triton_output_http]
)
request_grpc = grpc_client.infer(
    'fil',
    model_version='1',
    inputs=[triton_input_grpc],
    outputs=[triton_output_grpc]
)

# Get results as numpy arrays
result_http = request_http.as_numpy('output__0')
result_grpc = request_grpc.as_numpy('output__0')

# Check that we got the same result with both GRPC and HTTP
numpy.testing.assert_almost_equal(result_http, result_grpc)
```

## Modifications and Code Contributions
For full implementation details as well as information on modifying the FIL
backend code or contributing code to the project, please see
[CONTRIBUTING.md](https://github.com/wphicks/triton_fil_backend/blob/main/CONTRIBUTING.md).
