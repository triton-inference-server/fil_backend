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
Like all Triton backends, models deployed via the FIL backend make use of a
specially laid-out "model repository" directory containing at least one
serialized model and a `config.pbtxt` configuration file:
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
    value: { string_value: "xgboost_json" }
  },
  {
    key: "output_class"
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
Support](https://developers.google.com/protocol-buffers/docs/text-format-spec)
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

The FIL backend currently exclusively uses 32-bit precision. 64-bit model
parameters are rounded to 32-bit values, although optional support for 64-bit
execution should be added in the near future. At present, however, both
inputs and outputs should *always* use `TYPE_FP32` for their `data_type`.

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
Support](docs/model_support.md).

The `model_type` option is used to indicate which of these serialization
formats your model uses: `xgboost` for XGBoost binary, `xgboost_json` for
XGBoost JSON, `lightgbm` for LightGBM, or `treelite_checkpoint` for
Treelite:

```
parameters [
  {
    key: "model_type"
    value: { string_value: "xgboost_json" }
  }
]
```

#### Model Filenames
For each model type, Triton expects a particular default filename:
- `xgboost.model` for XGBoost Binary
- `xgboost.json` for XGBoost JSON
- `model.txt` for LightGBM
- `checkpoint.tl` for Treelite
It is recommended that you use these filenames, but custom filenames can be
specified using Triton's usual configuration options.

### Classification vs. Regression (`output_class`)
Set `output_class` to `true` if your model is a classification model or
`false` if your model is a regressor:
```
parameters [
  {
    key: "output_class"
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

#### `use_experimental_optimizations`
As of release 22.11, this flag only affects CPU deployments. Setting it to
true can significantly improve both latency and throughput. In the future,
the current experimental optimizations will become default, and new
performance optimizations will be trialed using this flag. Even these
experimental optimizations are thoroughly tested before release, but they
offer less of a stability guarantee than the default execution mode.
```
parameters [
  {
    key: "use_experimental_optimizations"
    value: { string_value: "true" }
  }
]
```

#### `threads_per_tree` (GPU only)
This parameter applies only to GPU deployments and determines the number
of consecutive CUDA threads used to evaluate a single tree. While
correctly tuning this value offers significant performance improvements,
it is very difficult to determine the optimal value *a priori*.

In general, servers under higher load or those receiving larger batches from
clients will benefit from a higher value. On the other hand, more
powerful GPUs typically see optimal performance with a somewhat lower value.

To find the optimal value for your deployment, test under realistic traffic
and experiment with powers of 2 from 1 to 32.
```
parameters [
  {
    key: "threads_per_tree"
    value: { string_value: "4" }
  }
]
```

#### `storage_type` (GPU only)
This parameter determines how trees are represented in device memory.
Choosing `DENSE` will consume more memory but may sometimes offer
performance benefits. Choosing `SPARSE` will consume less memory and may
perform as well as or better than `DENSE` for some models. Choosing `AUTO`
will apply a heuristic that defaults to `DENSE` unless it is obvious that
doing so will consume significantly more memory.

`SPARSE8` is an
experimental format which offers an even smaller memory footprint than
`SPARSE` and may offer better throughput/latency in some cases. You should
thoroughly test your model's output with `SPARSE8` if you choose to use it.

```
parameters [
  {
    key: "storage_type"
    value: { string_value: "SPARSE" }
  }
]
```

#### `algo` (GPU only)
This parameter determines how nodes within a tree are organized in memory
as well as how they are accessed during inference. `NAIVE` uses a
breadth-first layout and is the only value which should be used with the
`SPARSE` or `SPARSE8` storage types. `TREE_REORG` rearranges trees to improve
memory coalescing. `BATCH_TREE_REORG` is similar to `TREE_REORG` but
performs inference on several rows at once within a thread block.
`ALGO_AUTO` will default to `NAIVE` for sparse storage and
`BATCH_TREE_REORG` for dense storage.

Different settings for this parameter may produce modest performance
benefits depending on the details of the model.
```
parameters [
  {
    key: "algo"
    value: { string_value: "NAIVE" }
  }
]
```

#### `blocks_per_sm` (GPU only)
This experimental option attempts to launch the indicated number of blocks
per streaming multiprocessor if set to a value greater than 0. This will
fail if your hardware cannot support the number of threads associated with
this value. Tweaking this number can sometimes result in small
performance benefits.
```
parameters [
  {
    key: "blocks_per_sm"
    value: { string_value: "2" }
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
