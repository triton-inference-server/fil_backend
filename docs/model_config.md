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
```
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
Note that (as suggested by the extension), this is a [Protobuf text
file](https://developers.google.com/protocol-buffers/docs/text-format-spec)
and should be formatted accordingly.

## Specifying the backend
If you wish to use the FIL backend, you must indicate this in the
configuration file with the top-level `backend: "fil"` option. For
information on models supported by the FIL backend, see [Model
Support](https://developers.google.com/protocol-buffers/docs/text-format-spec)
or the [FAQ
notebook](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1:-What-can-I-deploy-with-the-FIL-backend?)

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
`parameters` section of the configuration. Due to a limitation in Triton's
configuration parsing, all backend-specific parameters are represented as
string values and converted when read.

### Model Type
The FIL backend accepts models in a
[number](https://nbviewer.org/github/triton-inference-server/fil_backend/blob/main/notebooks/faq/FAQs.ipynb#FAQ-1:-What-can-I-deploy-with-the-FIL-backend?) of serialization formats,
including XGBoost JSON and binary formats, LightGBM's text format, and
Treelite's checkpoint format. For more information, see [Model
Support](https://github.com/triton-inference-server/fil_backend/blob/main/docs/model_support.md).

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

