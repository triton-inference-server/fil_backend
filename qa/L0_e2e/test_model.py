# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

try:
    import cuml
except Exception:
    cuml = None
import numpy as np
import pytest
from rapids_triton import Client
import xgboost as xgb
import lightgbm as lgbm


@pytest.fixture(scope="session")
def client():
    """A RAPIDS-Triton client for submitting inference requests"""
    client = Client()
    client.wait_for_server(10)
    return client

@pytest.fixture(scope='session')
def model_repo(pytestconfig):
    """The path to the model repository directory"""
    return pytestconfig.getoption('repo')

@pytest.fixture(scope="session")
def skip_shap(pytestconfig):
    return pytestconfig.getoption("no_shap")


def generate_config(
    model_name,
    config_directory,
    n_features,
    output_dim,
    num_classes,
    predict_proba,
    output_class,
    instance_kind,
    model_format,
    output_shap = False,
    max_batch_size=2048,
    use_experimental_optimizations=True,
    storage_type="AUTO",
    threshold=0.5,
):
    # Add treeshap output to xgboost_shap model
    treeshap_output = ""
    if output_shap:
      treeshap_output_dim = num_classes if num_classes > 2 else 1
      if treeshap_output_dim == 1:
          treeshap_output_str = f"{n_features + 1}"
      else:
          treeshap_output_str = f"{treeshap_output_dim}, {n_features + 1}"
      treeshap_output = f"""
      ,{{
          name: "treeshap_output"
          data_type: TYPE_FP32
          dims: [ {treeshap_output_str} ]
      }}"""
    config = f"""name: "{model_name}"
backend: "fil"
max_batch_size: {max_batch_size}
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {output_dim} ]
  }}
 {treeshap_output}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "{model_format}" }}
  }},
  {{
    key: "predict_proba"
    value: {{ string_value: "{str(predict_proba).lower()}" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "{str(output_class).lower()}" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "{threshold}" }}
  }},
  {{
    key: "algo"
    value: {{ string_value: "ALGO_AUTO" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "{storage_type}" }}
  }},
  {{
    key: "blocks_per_sm"
    value: {{ string_value: "0" }}
  }},
  {{
    key: "use_experimental_optimizations"
    value: {{ string_value: "{str(use_experimental_optimizations).lower()}" }}
  }}
]

dynamic_batching {{ }}"""
    config_path = os.path.join(config_directory, 'config.pbtxt')

    with open(config_path, 'w') as config_file:
        config_file.write(config)


# convenience wrapper around client.predict
def predict(client, model_name, X, shared_mem=None):
    client.triton_client.load_model(model_name)
    config = client.get_model_config(model_name)
  
    output_sizes = {
        output.name: np.product(output.dims) * np.dtype('float32').itemsize * X.shape[0]
        for output in config.output
    }
    inputs = {
        input_.name: X for input_ in config.input
    }
    return client.predict(model_name,
        inputs, output_sizes, shared_mem=shared_mem
    )

# create directories for triton models and configuration
def get_directories(model_repo, model_name):
    dir = os.path.abspath(os.path.join(model_repo, model_name))
    model_dir = os.path.join(dir, '1')
    os.makedirs(model_dir, exist_ok=True)
    return dir, model_dir

# cleanup our models after each test
@pytest.fixture(autouse=True)
def run_around_tests(model_repo):
    yield
    for filename in os.listdir(model_repo):
      filepath = os.path.join(model_repo, filename)
      try:
          shutil.rmtree(filepath)
      except OSError:
          os.remove(filepath)

has_gpu = os.environ.get("CPU_ONLY", 0) == 0
available_instance_types = ["KIND_CPU", "KIND_GPU"] if has_gpu else ["KIND_CPU" ]

def get_xgb_classifier(n_features, num_class):
    rng = np.random.RandomState(9)
    training_params = {
        "tree_method": "hist",
        "max_depth": 15,
        "n_estimators": 10,
    }
    model = xgb.XGBClassifier(**training_params)

    return model.fit(rng.random((1000, n_features)), rng.randint(0, num_class, 1000))

@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 4])
def test_xgb_classification_model(client, model_repo, instance_kind, num_class, use_experimental_optimizations):
    return
    n_features = 100
    model = get_xgb_classifier(n_features, num_class)

    base_name = "xgboost_{}_class_{}".format(num_class, instance_kind)
    # Binary probability output
    # Use json
    dir, model_dir = get_directories(model_repo, base_name + "_proba_json")
    model.save_model(os.path.join(model_dir, 'xgboost.json'))

    generate_config(
        base_name + "_proba_json",
        dir,
        n_features,
        num_class,
        num_class,
        predict_proba=True,
        output_class=True, # FIL gets upset if this is not set
        instance_kind=instance_kind,
        model_format="xgboost_json",
        use_experimental_optimizations=use_experimental_optimizations
    )      

    # Class output
    dir, model_dir = get_directories(model_repo, base_name + "_class")
    model.save_model(os.path.join(model_dir, 'xgboost.model'))
    generate_config(
        base_name + "_class",
        dir,
        n_features,
        1,
        num_class,
        predict_proba=False,
        output_class=True,
        instance_kind=instance_kind,
        model_format="xgboost",
    ) 

    # Shap output
    dir, model_dir = get_directories(model_repo, base_name + "_shap")
    model.save_model(os.path.join(model_dir, 'xgboost.model'))
    generate_config(
        base_name + "_shap",
        dir,
        n_features,
        1,
        num_class,
        predict_proba=False,
        output_class=True,
        output_shap=True,
        instance_kind=instance_kind,
        model_format="xgboost",
    ) 
    # Wait for triton load
    time.sleep(2)
    
    rng = np.random.RandomState(10)
    X = rng.random((100,n_features)).astype(np.float32)
    # Add some NaNs
    X.ravel()[rng.choice(X.size, int(X.size*0.1), replace=False)] = np.nan

    result = predict(client, base_name + "_proba_json", X)
    np.testing.assert_allclose(result['output__0'],  model.predict_proba(X), rtol=1e-3, atol=1e-3)

    result = predict(client, base_name + "_class", X)
    np.testing.assert_allclose(result['output__0'].astype(int),  model.predict(X), rtol=1e-3, atol=1e-3)

    if instance_kind == "KIND_GPU":
        result = predict(client, base_name + "_shap", X)
        np.testing.assert_allclose(result['treeshap_output'],  model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(result['treeshap_output'].sum(axis=-1),  model.predict(X, output_margin=True), rtol=1e-3, atol=1e-3)
    
        # Test shmem
        result = predict(client, base_name + "_class", X, shared_mem='cuda')
        np.testing.assert_allclose(result['output__0'].astype(int),  model.predict(X), rtol=1e-3, atol=1e-3)

def get_xgb_regressor(n_features):
    rng = np.random.RandomState(11)
    training_params = {
        "tree_method": "hist",
        "max_depth": 15,
        "n_estimators": 10,
    }
    model = xgb.XGBRegressor(**training_params)

    return model.fit(rng.random((1000, n_features)), rng.random(1000))

@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_xgb_regression_model(client, model_repo, instance_kind, use_experimental_optimizations):
    return
    n_features = 100
    model = get_xgb_regressor(n_features)

    base_name = "xgboost_regression_{}".format(instance_kind)
    dir, model_dir = get_directories(model_repo, base_name)
    model.save_model(os.path.join(model_dir, 'xgboost.model'))

    generate_config(
        base_name,
        dir,
        n_features,
        1,
        1,
        predict_proba=False,
        output_class=False,
        instance_kind=instance_kind,
        model_format="xgboost",
        use_experimental_optimizations=use_experimental_optimizations
    )      

    # Shap output
    dir, model_dir = get_directories(model_repo, base_name + "_shap")
    model.save_model(os.path.join(model_dir, 'xgboost.model'))
    generate_config(
        base_name + "_shap",
        dir,
        n_features,
        1,
        1,
        predict_proba=False,
        output_class=False,
        output_shap=True,
        instance_kind=instance_kind,
        model_format="xgboost",
    ) 
    # Wait for triton load
    time.sleep(2)
    
    rng = np.random.RandomState(10)
    X = rng.random((100,n_features)).astype(np.float32)
    # Add some NaNs
    X.ravel()[rng.choice(X.size, int(X.size*0.1), replace=False)] = np.nan

    result = predict(client, base_name, X)
    np.testing.assert_allclose(result['output__0'],  model.predict(X), rtol=1e-3, atol=1e-3)

    if instance_kind == "KIND_GPU":
        result = predict(client, base_name + "_shap", X)
        np.testing.assert_allclose(result['treeshap_output'],  model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(result['treeshap_output'].sum(axis=-1),  model.predict(X), rtol=1e-3, atol=1e-3)


def get_lgbm_classifier(n_features, num_class):
    rng = np.random.RandomState(9)
    model = lgbm.LGBMClassifier()

    return model.fit(rng.random((1000, n_features)), rng.randint(0, num_class, 1000))

@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 10])
def test_lgbm_classification_model(client, model_repo, instance_kind, num_class, use_experimental_optimizations):
    n_features = 100
    model = get_lgbm_classifier(n_features, num_class)

    base_name = "lgbm{}_class_{}".format(num_class, instance_kind)
    # Binary probability output
    dir, model_dir = get_directories(model_repo, base_name + "_proba")
    model.booster_.save_model(os.path.join(model_dir, 'model.txt'))

    generate_config(
        base_name + "_proba",
        dir,
        n_features,
        num_class,
        num_class,
        predict_proba=True,
        output_class=True, # FIL gets upset if this is not set
        instance_kind=instance_kind,
        model_format="lightgbm",
        use_experimental_optimizations=use_experimental_optimizations
    )
    

    # Class output
    dir, model_dir = get_directories(model_repo, base_name + "_class")
    model.booster_.save_model(os.path.join(model_dir, 'model.txt'))
    generate_config(
        base_name + "_class",
        dir,
        n_features,
        1,
        num_class,
        predict_proba=False,
        output_class=True,
        instance_kind=instance_kind,
        model_format="lightgbm",
    ) 

    # Shap output
    dir, model_dir = get_directories(model_repo, base_name + "_shap")
    model.booster_.save_model(os.path.join(model_dir, 'model.txt'))
    generate_config(
        base_name + "_shap",
        dir,
        n_features,
        1,
        num_class,
        predict_proba=False,
        output_class=True,
        output_shap=True,
        instance_kind=instance_kind,
        model_format="lightgbm",
    ) 
    
    rng = np.random.RandomState(10)
    X = rng.random((100,n_features)).astype(np.float32)
    # Add some NaNs
    X.ravel()[rng.choice(X.size, int(X.size*0.1), replace=False)] = np.nan

    result = predict(client, base_name + "_proba", X)
    np.testing.assert_allclose(result['output__0'],  model.predict_proba(X), rtol=1e-3, atol=1e-3)

    result = predict(client, base_name + "_class", X)
    np.testing.assert_allclose(result['output__0'].astype(int),  model.predict(X), rtol=1e-3, atol=1e-3)

    if instance_kind == "KIND_GPU":
        result = predict(client, base_name + "_shap", X)
        np.testing.assert_allclose(result['treeshap_output'].sum(axis=-1),  model.predict(X, raw_score=True), rtol=1e-3, atol=1e-3)