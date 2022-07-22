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
frameworks (including XGBoost, LightGBM, Scikit-Learn, and cuML) to be deployed
in a [Triton inference
server](https://developer.nvidia.com/nvidia-triton-inference-server) using the
RAPIDS [Forest Inference
LIbrary](https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35)
for fast GPU-based inference. Using this backend, forest models can be deployed
seamlessly alongside deep learning models for fast, unified inference
pipelines.

## Shapley Value Support

This backend also provides the support to enable Shapley values for forest
models. Shapley values provide an estimate of feature contribution and
importance to explain inference. To enable Shapley value outputs, please see
the section on [Configuration](#configuration) for instructions.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Installation

#### Pre-built container

Pre-built Triton containers are available from NGC and may be pulled down via

```bash
docker pull nvcr.io/nvidia/tritonserver:22.06-py3
```

Note that the FIL backend cannot be used in the `21.06` version of this
container; the `21.06.1` patch release or later is required.

#### Building locally

To build the Triton server container with the FIL backend, you can invoke the
`build.sh` wrapper from the repo root:

```bash
./build.sh
```

#### Customized builds (ADVANCED)

**This build option has been removed in Triton 21.11 but may be re-introduced
at a later date. Please file an issue if you have need for greater build
customization than is provided by standard build options.**

#### Docker-free build (EXPERIMENTAL)

To build the FIL backend library on the host (as opposed to in Docker), the
following prerequisites are required:
- cMake >= 3.21, != 3.23
- ccache
- ninja
- RapidJSON
- nvcc >= 11.0
- The CUDA Toolkit
All of these except nvcc and the CUDA Toolkit can be installed in a `conda`
environment via the following:
```bash
conda env create -f conda/environments/rapids_triton_dev.yml
```

Then the backend library can be built within this environment as follows:
```bash
conda activate rapids_triton_dev
./build.sh --host server
```
The backend libraries will be installed in `install/backends/fil` and can be
copied to the backends directory of a Triton installation.

This build path is considered experimental. It is primarily used for rapid
iteration during development. Building a full Triton Docker image is the
recommended method for environments that require the latest FIL backend code.

### Running the server

Before starting the server, you will need to set up a "model repository"
directory containing the model you wish to serve as well as a configuration
file. The FIL backend currently supports forest models serialized in XGBoost's
binary format, XGBoost's JSON format, LightGBM's text format, and Treelite's
binary checkpoint format. For those using cuML or Scikit-Learn random forest
models, please see the [documentation on how to prepare such
models](https://github.com/triton-inference-server/fil_backend/blob/main/SKLearn_and_cuML.md)
for use in Triton.

**NOTE: XGBoost 1.6 introduced a change in the XGBoost JSON serialization
format. This change will be supported in Triton 22.07. Earlier versions of
Triton will NOT support JSON-serialized models from XGBoost>=1.6.**

Once you have a serialized model, you will need to prepare a
directory structure similar to the following example, which uses an XGBoost
binary file:

```
model_repository/
`-- fil
    |-- 1
    |   `-- xgboost.model
    `-- config.pbtxt
```

By default, the FIL backend assumes that XGBoost binary models will be named
`xgboost.model`, XGBoost json models will be named `xgboost.json`, LightGBM
models will be named `model.txt`, and Treelite binary models will be named
`checkpoint.tl`, but this can be tweaked through standard Triton configuration
options.

The FIL backend repository includes [a Python
script](https://github.com/triton-inference-server/fil_backend/blob/main/qa/L0_e2e/generate_example_model.py)
for generating example models and configuration files using XGBoost, LightGBM,
Scikit-Learn, and cuML. These examples may serve as a useful template for
setting up your own models on Triton. See the [documentation on generating
example
models](https://github.com/triton-inference-server/fil_backend/blob/main/Example_Models.md)
for more details.

#### Configuration

Once you have chosen a model to deploy and placed it in the correct directory
structure, you will need to create a corresponding `config.pbtxt` file. An
example of this configuration file is shown below:

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
  },
  {
    key: "threads_per_tree"
    value: { string_value: "1" }
  },
  {
    key: "transfer_threshold"
    value: { string_value: "0" }
  },
  {
    key: "use_experimental_optimizations"
    value: { string_value: "false" }
  }
]

dynamic_batching { }
```

**NOTE:** At this time, the FIL backend supports **only** `TYPE_FP32` for input
and output. Attempting to use any other type will result in an error.

To enable Shapley value outputs, modify the `output` section of the above example
to look like:

```
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
```

**NOTE:** The dimensions of Shapley value outputs for a multi-class problem
are of the form `[ n_classes, (n_features + 1) ]`, while for a binary-class problem
they are of the form `[ n_features + 1 ]`. The additional column in the output stores 
the bias term. At this moment, Shapley value support is only available when this
backend is GPU enabled.

For a full description of the configuration schema, see the Triton [server
docs](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).
Here, we will simply summarize the most commonly-used options and those
specific to FIL:

- `max_batch_size`: The maximum number of samples to process in a batch. In
  general, FIL's efficient handling of even large forest models means that this
  value can be quite high (2^13 in the example), but this may need to be
  reduced for your particular hardware configuration if you find that you are
  exhausting system resources (such as GPU or system RAM).
- `input`: This configuration block specifies information about the input
  arrays that will be provided to the FIL model. The `dims` field should be set
  to `[ NUMBER_OF_FEATURES ]`, but all other fields should be left as they
  are in the example. Note that the `name` field should always be given a value
  of `input__0`. Unlike some deep learning frameworks where models may have
  multiple input layers with different names, FIL-based tree models take a
  single input array with a consistent name of `input__0`.
- `output`: This configuration block specifies information about the arrays
  output by the FIL model. If the `predict_proba` option (described later) is
  set to "true" and you are using a classification model, the `dims` field
  should be set to `[ NUMBER_OF_CLASSES ]`. Otherwise, this can simply be `[ 1 ]`,
  indicating that the model returns a single class ID for each sample.
- `instance_group`: This setting determines whether inference will take place
  on the GPU (`KIND_GPU`) or CPU (`KIND_CPU`)
- `cpu_nthread`: The number of threads to use when running prediction on the CPU.
- `parameters`: This block contains FIL-specific configuration details. Note
  that all parameters are input as strings and should be formatted with `key`
  and `value` fields as shown in the example.
  * `model_type`: One of `"xgboost"`, `"xgboost_json"`, `"lightgbm"`, or
    `"treelite_checkpoint"`, indicating whether the provided model is in
    XGBoost binary format, XGBoost JSON format, LightGBM text format, or
    Treelite binary format respectively.
  * `predict_proba`: Either `"true"` or `"false"`, depending on whether the
    desired output is a score for each class or merely the predicted class ID.
  * `output_class`: Either `"true"` or `"false"`, depending on whether the
    model is a classification or regression model.
  * `threshold`: The threshold score used for class prediction.
  * `algo`: One of `"ALGO_AUTO"`, `"NAIVE"`, `"TREE_REORG"` or
    `"BATCH_TREE_REORG"` indicating which FIL inference algorithm to use. More
    details are available in the [cuML
    documentation](https://docs.rapids.ai/api/cuml/stable/api.html?highlight=algo_t#cuml.ForestInference.load_from_treelite_model).
    If you are uncertain of what algorithm to use, we recommend selecting
    `"ALGO_AUTO"`, since it is a safe choice for all models.
  * `storage_type`: One of `"AUTO"`, `"DENSE"`, `"SPARSE"`, and `"SPARSE8"`, indicating
    the storage format that should be used to represent the imported model.
    `"AUTO"` indicates that the storage format should be automatically chosen.
    `"SPARSE8"` is currently experimental.
  * `blocks_per_sm`: If set to any nonzero value (generally between 2 and 7),
    this provides a limit to improve the cache hit rate for large forest
    models. In general, network latency will significantly overshadow any
    speedup from tweaking this setting, but it is provided for cases where
    maximizing throughput is essential. Please see the [cuML
    documentation](https://docs.rapids.ai/api/cuml/stable/api.html?highlight=algo_t#cuml.ForestInference.load_from_treelite_model)
    for a more thorough explanation of this parameter and how it may be used.
  * `threads_per_tree`: Determines number of threads used to use for inference
    on a single tree. Increasing this above 1 can improve memory bandwidth near
    the tree root but use more shared memory. In general, network latency will
    significantly overshadow any speedup from tweaking this setting, but it is
    provided for cases where maximizing throughput is essential.  for a more
    thorough explanation of this parameter and how it may be used.
  * `transfer_threshold`: If the number of samples in a batch exceeds this
    value and the model is deployed on the GPU, then GPU inference will be
    used. Otherwise, CPU inference will be used for batches received in host
    memory with a number of samples less than or equal to this threshold.  For
    most models and systems, the default transfer threshold of 0 (meaning that
    data is always transferred to the GPU for processing) will provide optimal
    latency and throughput, but for low-latency deployments with the
    `use_experimental_optimizations` flag set to `true`, higher values may be
    desirable.
  * `use_experimental_optimizations`: Triton 22.04 introduces a new CPU
    optimization mode which can significantly improve both latency and
    throughput. For low-latency deployments in particular, the throughput
    improvement may be substantial. Due to the relatively recent development of
    this approach, it is still considered experimental, but it can be enabled
    by setting this flag to `true`. Later releases will make this execution
    mode the default and deprecate this flag. See
    [below](#experimental-cpu-optimizations-in-2204) for more information.
- `dynamic_batching`: This configuration block specifies how Triton should
  perform dynamic batching for your model. Full details about these options can
  be found in the main [Triton
  documentation](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#models-and-schedulers). You may find it useful to test your configuration using the [Triton `perf_analyzer` tool](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md) in order to optimize performance.
  * `max_queue_delay_microseconds`: How long of a window in which requests can
    be accumulated to form a batch.

Note that the configuration is in protobuf format. If invalid protobuf is
provided, the model will fail to load, and you will see an error line in the
server log containing `Error parsing text-format inference.ModelConfig:`
followed by the line and column number where the parsing error occurred.

#### CPU-only Execution
While most users will want to take advantage of the higher throughput and lower
latency that can be achieved through inference on a GPU, it is possible to
configure a model to perform inference on the CPU for e.g. testing a deployment
on a machine without a GPU. This is controlled on a per-model basis via the
`instance_group` configuration option described above.

**WARNING:** Triton allows clients running on the same machine as the server to
submit inference requests using CUDA IPC via Triton's [CUDA shared memory
mode](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md).
While it is possible to submit requests in this manner to a model running on
the CPU, an intermittent bug in versions 21.07 and earlier can occasionally
cause incorrect results to be returned in this situation. It is generally
recommended that Triton's CUDA shared memory mode **not** be used to submit
requests to CPU-only models for FIL backend versions before 21.11.

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
[here](https://github.com/triton-inference-server/client/#triton-client-libraries-and-examples).
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

# Generate example data to classify
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

##### Categorical Feature Support
As of version 21.11, the FIL backend includes support for models with
categorical features (e.g. some [LightGBM
models](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support)).
These models can be deployed just like any other model, but it is worth
remembering that (as with any other inference pipeline which includes
categorical features), care must be taken to ensure that the categorical
encoding used during inference matches that used during training. If the data
passed through at inference time does not contain all of the categories used
during training, there is no way to reconstruct the correct mapping of
features, so some record must be made of the complete set of categories used
during training. With that record, categorical columns can be appropriately
converted to float32 columns, and the data can be sent to the server as shown
in the above client example.

##### Experimental CPU Optimizations in 22.04
As described above in configuration options, a new CPU execution mode was
introduced in Triton 22.04. This mode offers significantly improved latency and
throughput for CPU evaluation, especially for low-latency deployment
configurations. When a model is deployed on GPU, turning on this execution mode
can still provide some benefit if `transfer_threshold` is set to any value
other than 0. In this case, when the server is under a light enough load to
keep server-side batch sizes under this threshold, it will take advantage of
the newly-optimized CPU execution mode to keep latency as low as possible while
maximizing throughput. If server load increases, the FIL backend will
automatically scale up onto the GPU to maintain optimal performance. The
optimal value of `transfer_threshold` will depend on the available hardware and
the size of the model, so some testing may be required to find the best
configuration.

This mode is still considered experimental in release 22.04, so it must be
explicitly turned on by setting `use_experimental_optimizations` to `true` in
the model's `config.pbtxt`.

## Modifications and Code Contributions
For full implementation details as well as information on modifying the FIL
backend code or contributing code to the project, please see
[CONTRIBUTING.md](https://github.com/wphicks/triton_fil_backend/blob/main/CONTRIBUTING.md).
