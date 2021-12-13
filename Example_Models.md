# Generating Example Models

The FIL backend's testing infrastructure includes [a
script](https://github.com/triton-inference-server/fil_backend/blob/main/qa/L0_e2e/generate_example_model.py)
for generating example models, putting them in the correct directory layout,
and generating an associated config file. This can be helpful both for
providing a template for your own models and for testing your Triton
deployment.

## Prerequisites
To use the model generation script, you will need to install
[cuML](https://rapids.ai/start.html#rapids-release-selector) and whatever
forest model framework you wish to use
([LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html),
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html), or
[Scikit-Learn](https://scikit-learn.org/stable/install.html)). For convenience,
a Conda environment [config
file](https://github.com/triton-inference-server/fil_backend/blob/main/conda/environments/triton_test.yml)
is included in the FIL backend repo which can be used to install all of these
frameworks:

```bash
git clone https://github.com/triton-inference-server/fil_backend.git
cd fil_backend
conda env create -f conda/environments/triton_test.yml
conda activate triton_test
```

## Usage

The simplest possible invocation of the example generation script is just:

```bash
python  qa/L0_e2e/generate_example_model.py
```

This will create an example XGBoost model, serialize it to XGBoost's binary
format and store it (with full configuration) within the
`qa/L0_e2e/model_repository` directory.

### Arguments
You can provide additional arguments to the model generation script to control
all details of the generated model. Available arguments are described in the
following sections.

#### Model framework
- `--type`: Takes one of `lightgbm`, `xgboost`, `sklearn` or `cuml` as argument
  and determines what framework will be used to train the model. Defaults to
  `xgboost`.
- `--format`: Determines what format to serialize the model to for frameworks
  which support multiple serialization formats. One of `xgboost`,
  `xgboost_json`, `lightgbm`, or `pickle`. If omitted, this will default to a
  valid choice for the chosen framework.

#### Model metadata
- `--name`: An arbitrary string used to identify the generated model. If
  omitted, a string will be generated from the model type, serialization
  format, and task.
- `--repo`: Path to the directory where you wish to set up your model
  repository. This argument is required if this script is invoked outside of
  the FIL backend Git repository. If omitted, it will default to
  `qa/L0_e2e/model_repository` from the Git repository root.

#### Model details
- `--task`: One of `classification` or `regression` indicating the type of
  inference task for this model.
- `--depth`: The maximum depth for trees in this model.
- `--trees`: The maximum number of trees in this model.
- `--classes`: The number of classes for classification models.
- `--features`: The number of features used for each sample.
- `--samples`: The number of randomly-generated samples to use when training
  the example model.
- `--threshold`: The threshold for classification decisions in classifier
  models.
- `--predict_proba`: A flag indicating that class scores should be outputted
  instead of class IDs for classifiers.

#### Triton server controls
- `--batching_window`: Maximum time in microseconds for Triton to spend
  gathering samples for a single batch

### SKLearn and cuML models
Note that this example script generates only the model pickle file for
Scikit-Learn and cuML models. These must be converted to Treelite checkpoints
as described in the [documentation for using these
frameworks](https://github.com/triton-inference-server/fil_backend.git). An
example invocation for Scikit-Learn is shown below:

```bash
python  qa/L0_e2e/generate_example_model.py --type sklearn --name skl_example
./scripts/convert_sklearn qa/L0_e2e/model_repository/skl_example/1/model.pkl
```
## Testing example models
Once you have generated an example model (or set up a real model), you can test
it using the `qa/L0_e2e/test_model.py` script. After [starting the
server](https://github.com/triton-inference-server/fil_backend#starting-the-server),
the simplest invocation of this script is just:
```bash
python qa/L0_e2e/test_model.py --name $NAME_OF_MODEL
```
This will run a number of randomly-generated samples through your model both in
Triton and locally. The results will be compared to ensure they are the same.
At the end of the run, some throughput and latency numbers will be printed to
the terminal, but please note that these numbers are **not indicative of
real-world throughput and latency performance**. This script is designed to
rigorously test unlikely corner cases in ways which will hurt reported
performance. The output statistics are provided merely to help catch
performance regressions between different versions or deployments of Triton and
are meaningful only when compared to other test runs with the same parameters.
To get an accurate picture of model throughput and latency, use Triton's [Model
Analyzer](https://github.com/triton-inference-server/model_analyzer) which
includes an easy-to-use tool for meaningfully testing model performance.

### Additional arguments

- `--name`: The name of the model to test.
- `--repo`: The path to the model repository. If this script is not invoked
  from within the FIL backend Git repository, this option must be specified. It
  defaults to `qa/L0_e2e/model_repository`.
- `--host`: The URL for the Triton server. Defaults to `localhost`.
- `--http_port`: If using a non-default HTTP port for Triton, the correct port
  can be specified here.
- `--grpc_port`: If using a non-default GRPC port for Triton, the correct port
  can be specified here.
- `--protocol`: While the test script will do brief tests of both HTTP and
  GRPC, the specified protocol will be used for more intensive testing.
- `--samples`: The total number of samples to test for each batch size
  provided. Defaults to 8192.
- `--batch_size`: This argument can take an arbitrary number of values. For
  each provided value, all samples will be broken down into batches of the
  given size and the model will be evaluated against all such batches.
- `--shared_mem`: This argument can take up to two values. These values can be
  either `None` or `cuda` to indicate whether the tests should use no shared
  memory or CUDA shared memory. If both are given, tests will alternate between
  the two. Defaults to both.
- `--concurrency`: The number of concurrent threads to use for generating
  requests. Higher values will provide a more rigorous test of the server's
  operation when processing many simultaneous requests.
- `--timeout`: The longest to wait for all samples to be processed for a
  particular batch size. The appropriate value depends on your hardware,
  networking configuration, and total number of samples.
- `--retries`: The number of times to retry requests in order to handle network
  failures.
  can be specified here.
