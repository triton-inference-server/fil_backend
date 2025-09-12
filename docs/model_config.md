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
# FIL Backend Model Configuration
Like all Triton
[backends](https://github.com/triton-inference-server/backend/#where-can-i-find-all-the-backends-that-are-available-for-triton),
models deployed via the FIL backend make use of a specially laid-out "model
repository" directory containing at least one serialized model and a
`config.pbtxt` configuration file:
```
model_repository/
├─ example/
│  ├─ 1/
│  │  ├─ model.json
│  ├─ 2/
│  │  ├─ model.json
│  ├─ config.pbtxt
```

Documentation for general Triton configuration options is available in the
[main Triton
docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration),
but here, we will review options specific to the FIL backend. For a more
succinct overview, refer to the [FAQ notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb), which includes guides for configuration under specific deployment scenarios.

## Structure of configuration file
A typical `config.pbtxt` file might look something like this:
```protobuf
backend: "fil"
max_batch_size: 32768
input [
 {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 32 ]
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
    value: { string_value: "xgboost_ubj" }
  },
  {
    key: "is_classifier"
    value: { string_value: "true" }
  }
]

dynamic_batching {}
```
Note that (as suggested by the file extension), this is a [Protobuf text file](https://developers.google.com/protocol-buffers/docs/text-format-spec)
and should be formatted accordingly.

## Specifying the backend
If you wish to use the FIL backend, you must indicate this in the
configuration file with the top-level `backend: "fil"` option. For
information on models supported by the FIL backend, see [Model
Support](model_support.md)
or the [FAQ
notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1:-What-can-I-deploy-with-the-FIL-backend?).

## Maximum Batch Size
Because of the relatively quick execution speed of most tree models and the
inherent parallelism of FIL, typically the only limitation on the maximum batch
size is the size of Triton's CUDA memory pool [set at
launch](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-4.2-How-do-I-increase-Triton's-device-memory-pool?).
Nevertheless, you must specify *some* maximum batch size here, so setting it
to whatever (large) value is consistent with your memory usage needs
alongside other models should work fine:
```
max_batch_size: 1048576
```

## Inputs and Outputs
Input and output tensors for a model are described using three entries:
`name`, `data_type`, and `dims`. The `name` for the input tensor should
*always* be `"input__0"`, and the name for the primary output tensor should
*always* be `"output__0"`.

The FIL backend currently
[exclusively](https://github.com/triton-inference-server/fil_backend/issues/231)
uses 32-bit precision. 64-bit model parameters are rounded to 32-bit values,
although optional support for 64-bit execution should be added in the near
future. At present, however, both inputs and outputs should *always* use
`TYPE_FP32` for their `data_type`.

The dimensions of the I/O tensors do *not* include the batch dimension and
are model-dependent. The input tensor's single dimension should just be the
number of features (columns) in a single input sample (row).

The output tensor's dimensions depend on whether the `predict_proba`
option (described below) is used or not. If it is not used, the return value
for each sample is just the index of the output class. In this case, the
single output dimension is just `1`. Otherwise, a probability will be returned for every class in the model and the single output dimension should be the number of classes. Binary models are considered to have a single class for data transfer efficiency.

Below, we see an example I/O specification for a model with 32 input
features and 3 output classes with the `predict_proba` flag enabled:
```
input [
 {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 32 ]
  }
]
output [
 {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }
]
```

## Specifying CPU/GPU execution
Triton loads multiple copies or "instances" of its models to help take
maximum advantage of available hardware. You may control details of this via
the top-level `instance_group` option.

The simplest use of this option is simply:
```
instance_group [{ kind: KIND_AUTO }]
```
This will load one instance on each available NVIDIA GPU. If no compatible
NVIDIA GPU is found, a single instance will instead be loaded on the CPU.
`KIND_GPU` and `KIND_CPU` are used in place of `KIND_AUTO` if you wish to
explicitly specify GPU or CPU execution.

You may also specify `count` in addition to `kind` if you wish to load
additional model instances. See the [main Triton docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration) for more information.

## Dynamic Batching
One of the most useful features of the Triton server is its ability to batch
multiple requests and evaluate them together. To enable this feature,
include the following top-level option:
```
dynamic_batching {}
```
Except for cases where latency is *critical* down to the level of
microseconds, we strongly recommend enabling dynamic batching for all
deployments.

## FIL-Specific Options
All other options covered here are specific to FIL and will go in the
`parameters` section of the configuration. Triton's backend-specific parameters
are represented as string values and converted when read.

### Model Type
The FIL backend accepts models in a
[number](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1:-What-can-I-deploy-with-the-FIL-backend?) of serialization formats,
including XGBoost JSON and binary formats, LightGBM's text format, and
Treelite's checkpoint format. For more information, see [Model
Support](model_support.md).

The `model_type` option is used to indicate which of these serialization
formats your model uses: `xgboost_ubj` for XGBoost UBJSON, `xgboost_json` for
XGBoost JSON, `xgboost` for XGBoost binary (legacy), `lightgbm` for LightGBM,
or `treelite_checkpoint` for Treelite:

```
parameters [
  {
    key: "model_type"
    value: { string_value: "xgboost_ubj" }
  }
]
```

#### Model Filenames
For each model type, Triton expects a particular default filename:
- `xgboost.ubj` for XGBoost UBJSON
- `xgboost.json` for XGBoost JSON
- `xgboost.model` for XGBoost Binary (Legacy)
- `model.txt` for LightGBM
- `checkpoint.tl` for Treelite
It is recommended that you use these filenames, but custom filenames can be
specified using Triton's usual
[configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration)
options.

### Classification vs. Regression (`is_classifier`)
Set `is_classifier` to `true` if your model is a classification model or
`false` if your model is a regressor:
```
parameters [
  {
    key: "is_classifier"
    value: { string_value: "true" }
  }
]
```

### Classification confidence scores (`predict_proba`)
For classifiers, if you wish to return a confidence score for each class
rather than simply a class ID, set `predict_proba` to `true`
```
parameters [
  {
    key: "predict_proba"
    value: { string_value: "true" }
  }
]
```

### Allow unknown fields in XGBoost JSON model (`xgboost_allow_unknown_field`)
For XGBoost JSON models, ignore unknown fields instead of throwing a validation
error. This flag is ignored for other kinds of models.
```
parameters [
  {
    key: "xgboost_allow_unknown_field"
    value: { string_value: "true" }
  }
]
```

### Decision Threshold
For binary classifiers, it is sometimes helpful to set a specific
confidence threshold for positive decisions. This can be set via the
`threshold` parameter. If unset, an implicit threshold of 0.5 is used for
binary classifiers.
```
parameters [
  {
    key: "threshold"
    value: { string_value: "0.3" }
  }
]
```

### Performance parameters
The FIL backend includes several parameters that can be tuned to optimize
latency and throughput for your model. These parameters will not affect model
output, but experimenting with them can significantly improve model
performance for your specific use case.

#### `chunk_size` (GPU only)
This parameter applies only to GPU deployments and determines how batches are
further subdivided for parallel processing. The optimal value depends on hardware,
model, and batch size, and it is difficult to predict in advance.

In general, servers under higher load or those receiving larger batches from
clients will benefit from a higher value. `chunk_size` can be any power of 2
between 1 and 32.
```
parameters [
  {
    key: "chunk_size"
    value: { string_value: "4" }
  }
]
```

#### `layout` (GPU only)
This parameter determines how nodes within a tree are organized in memory
as well as how they are accessed during inference.

The `depth_first` and `breadth_first` options lay out the nodes in the
depth-first and breadth-first orders, respectively.

The `layered` option is unique,
in that it interleaves nodes from multiple trees, unlike `depth_first`
or `breadth_first`. When `layered` option is chosen, FIL will traverse
the forest by proceeding through the root nodes of each tree first,
followed by the children of those root nodes for each tree,
and so forth. This traversal order ensures that all nodes of a
particular tree at a particular depth are traversed together.

In general, `depth_first` tends to work the best for deeper trees
(depth 4 or greater), whereas `breadth_first` tends to work better
for shallow trees. Your mileage  may vary, so make sure to measure
performance to choose the best configuration.

```
parameters [
  {
    key: "layout"
    value: { string_value: "depth_first" }
  }
]
```

#### `transfer_threshold` (GPU only)
For extremely lightweight models operating on a deployment with very light
traffic and small batch sizes, the overhead of moving data to the GPU
sometimes outweighs the faster inference. As traffic increases, however,
you will want to take advantage of available NVIDIA GPUs to provide optimal
throughput/latency. To facilitate this, the `transfer_threshold` can be set to
some integer value indicating the number of rows beyond which data should
be transferred to the GPU. If this setting is beneficial at all, it
typically takes on a small value (~1-5 for typical hardware
configurations). Most models are unlikely to benefit from setting this to any
value other than 0, however.
```
parameters [
  {
    key: "transfer_threshold"
    value: { string_value: "2" }
  }
]
```
