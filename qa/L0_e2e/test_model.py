# Copyright (c) 2023, NVIDIA CORPORATION.
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
from functools import lru_cache

import numpy as np
import pytest
from rapids_triton import Client
import xgboost as xgb
import lightgbm as lgbm
import sklearn.ensemble
import treelite


@pytest.fixture(scope="session")
def client():
    """A RAPIDS-Triton client for submitting inference requests"""
    client = Client()
    client.wait_for_server(10)
    return client


@pytest.fixture(scope="session")
def model_repo(pytestconfig):
    """The path to the model repository directory"""
    return pytestconfig.getoption("repo")


@pytest.fixture(scope="session")
def skip_shap(pytestconfig):
    return pytestconfig.getoption("no_shap")


def generate_config(
    model_name,
    model_repo,
    num_features,
    output_dim,
    num_classes,
    predict_proba,
    output_class,
    instance_kind,
    model_format,
    output_shap=False,
    shap_num_outputs=1,
    max_batch_size=2048,
    use_experimental_optimizations=True,
    storage_type="AUTO",
    threshold=0.5,
):
    # Add treeshap output to xgboost_shap model
    treeshap_output = ""
    if output_shap:
        if shap_num_outputs == 1:
            treeshap_output_str = f"{num_features + 1}"
        else:
            treeshap_output_str = f"{shap_num_outputs}, {num_features + 1}"
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
    dims: [ {num_features} ]
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
    config_directory = os.path.abspath(os.path.join(model_repo, model_name))
    config_path = os.path.join(config_directory, "config.pbtxt")

    with open(config_path, "w") as config_file:
        config_file.write(config)


# convenience wrapper around client.predict
def predict(client, model_name, X, shared_mem=None, config=None):
    client.triton_client.load_model(model_name, config=config)
    config = client.get_model_config(model_name)

    output_sizes = {
        output.name: np.product(output.dims) * np.dtype("float32").itemsize * X.shape[0]
        for output in config.output
    }
    inputs = {input_.name: X for input_ in config.input}
    return client.predict(model_name, inputs, output_sizes, shared_mem=shared_mem)


# create directories for triton models and configuration
def get_model_directory(model_repo, model_name):
    dir = os.path.abspath(os.path.join(model_repo, model_name))
    model_dir = os.path.join(dir, "1")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


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

available_instance_types = ["KIND_CPU", "KIND_GPU"] if has_gpu else ["KIND_CPU"]


def run_classification_model(
    client,
    model_name,
    model_repo,
    model_format,
    X,
    instance_kind,
    num_class,
    num_features,
    use_experimental_optimizations,
    expected_class,
    expected_proba,
    expected_shap_sum,
):
    # probability output
    if expected_proba is not None:
        generate_config(
            model_name,
            model_repo,
            num_features,
            num_class,
            num_class,
            predict_proba=True,
            output_class=True,  # FIL gets upset if this is not set
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
        )

        result = predict(client, model_name, X)
        np.testing.assert_allclose(
            result["output__0"], expected_proba, rtol=1e-3, atol=1e-3
        )

    # Class output
    # Also test shmem
    generate_config(
        model_name,
        model_repo,
        num_features,
        1,
        num_class,
        predict_proba=False,
        output_class=True,
        instance_kind=instance_kind,
        model_format=model_format,
        use_experimental_optimizations=use_experimental_optimizations,
    )
    shared_mem = "cuda" if instance_kind == "KIND_GPU" else None
    result = predict(client, model_name, X, shared_mem=shared_mem)
    np.testing.assert_equal(result["output__0"], expected_class)

    # Shap output
    if instance_kind == "KIND_GPU" and expected_shap_sum is not None:
        shap_num_outputs = (
            1 if len(expected_shap_sum.shape) == 1 else expected_shap_sum.shape[1]
        )
        generate_config(
            model_name,
            model_repo,
            num_features,
            1,
            num_class,
            predict_proba=False,
            output_class=True,
            output_shap=True,
            shap_num_outputs=shap_num_outputs,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
        )
        result = predict(client, model_name, X)
        shap_sum = result["treeshap_output"].sum(axis=-1)
        np.testing.assert_allclose(
            shap_sum,
            expected_shap_sum.reshape(shap_sum.shape),
            rtol=1e-3,
            atol=1e-3,
        )


def run_regression_model(
    client,
    model_name,
    model_repo,
    model_format,
    X,
    instance_kind,
    num_features,
    use_experimental_optimizations,
    expected,
):
    # Binary probability output
    generate_config(
        model_name,
        model_repo,
        num_features,
        1,
        1,
        predict_proba=False,
        output_class=False,
        instance_kind=instance_kind,
        model_format=model_format,
        use_experimental_optimizations=use_experimental_optimizations,
    )

    result = predict(client, model_name, X)
    np.testing.assert_allclose(result["output__0"], expected, rtol=1e-3, atol=1e-3)

    # Shap output
    if instance_kind == "KIND_GPU":
        generate_config(
            model_name,
            model_repo,
            num_features,
            1,
            1,
            predict_proba=False,
            output_class=False,
            output_shap=True,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
        )
        result = predict(client, model_name, X)
        np.testing.assert_allclose(
            result["treeshap_output"].sum(axis=-1), expected, rtol=1e-3, atol=1e-3
        )


def data_with_categoricals(n_rows, n_cols, seed=23):
    rng = np.random.RandomState(seed)
    X = rng.random((n_rows, n_cols))
    # Add some NaNs
    X.ravel()[rng.choice(X.size, int(X.size * 0.1), replace=False)] = np.nan
    # Add categorical in column 1
    X[:, 1] = rng.randint(0, 5)
    return X.astype(np.float32)


@lru_cache
def get_xgb_classifier(num_features, num_class):
    rng = np.random.RandomState(9)
    feature_types = ["q"] * num_features
    feature_types[1] = "c"
    training_params = {
        "tree_method": "hist",
        "max_depth": 15,
        "n_estimators": 10,
        "feature_types": feature_types,
    }
    model = xgb.XGBClassifier(**training_params)

    return model.fit(
        data_with_categoricals(1000, num_features, 91), rng.randint(0, num_class, 1000)
    )


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 4])
def test_xgb_classification_model(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_xgb_classifier(num_features, num_class)
    base_name = "xgboost_{}_class_{}".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    model.save_model(os.path.join(model_dir, "xgboost.model"))
    X = data_with_categoricals(100, num_features, 13)
    run_classification_model(
        client,
        base_name,
        model_repo,
        "xgboost",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        model.predict_proba(X),
        model.predict(X, output_margin=True),
    )


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 4])
def test_xgb_classification_model_json(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_xgb_classifier(num_features, num_class)
    base_name = "xgboost_{}_class_{}_json".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    model.save_model(os.path.join(model_dir, "xgboost.json"))
    X = data_with_categoricals(100, num_features, 13)
    run_classification_model(
        client,
        base_name,
        model_repo,
        "xgboost_json",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        model.predict_proba(X),
        model.predict(X, output_margin=True),
    )


@lru_cache
def get_xgb_regressor(num_features):
    rng = np.random.RandomState(11)
    feature_types = ["q"] * num_features
    feature_types[1] = "c"
    training_params = {
        "tree_method": "hist",
        "max_depth": 15,
        "n_estimators": 10,
        "feature_types": feature_types,
    }
    model = xgb.XGBRegressor(**training_params)
    return model.fit(data_with_categoricals(1000, num_features, 13), rng.random(1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_xgb_regression_model(
    client, model_repo, instance_kind, use_experimental_optimizations
):
    num_features = 50
    model = get_xgb_regressor(num_features)

    base_name = "xgboost_regression_{}".format(instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    model.save_model(os.path.join(model_dir, "xgboost.model"))

    X = data_with_categoricals(100, num_features, 17)
    run_regression_model(
        client,
        base_name,
        model_repo,
        "xgboost",
        X,
        instance_kind,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
    )


@lru_cache
def get_lgbm_classifier(num_features, num_class):
    rng = np.random.RandomState(1132)
    model = lgbm.LGBMClassifier(n_estimators=20, categorical_feature=[1])
    return model.fit(
        data_with_categoricals(1000, num_features, 291), rng.randint(0, num_class, 1000)
    )


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 5])
def test_lgbm_classification_model(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_lgbm_classifier(num_features, num_class)
    base_name = "lgbm_{}_class_{}".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    model.booster_.save_model(os.path.join(model_dir, "model.txt"))
    X = data_with_categoricals(100, num_features, 13)
    run_classification_model(
        client,
        base_name,
        model_repo,
        "lightgbm",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        model.predict_proba(X),
        model.predict(X, raw_score=True),
    )


@lru_cache
def get_lgbm_regressor(num_features):
    rng = np.random.RandomState(1132)
    model = lgbm.LGBMRegressor(n_estimators=20, categorical_feature=[1])
    return model.fit(data_with_categoricals(1000, num_features, 291), rng.random(1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_lgbm_regression_model(
    client, model_repo, instance_kind, use_experimental_optimizations
):
    num_features = 50
    model = get_lgbm_regressor(num_features)

    base_name = "lgbm{}_reg".format(instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    model.booster_.save_model(os.path.join(model_dir, "model.txt"))
    X = data_with_categoricals(100, num_features, 72)
    run_regression_model(
        client,
        base_name,
        model_repo,
        "lightgbm",
        X,
        instance_kind,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
    )


@lru_cache
def get_sklearn_rf_classifier(num_features, num_class):
    rng = np.random.RandomState(1132)
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
    return model.fit(rng.random((1000, num_features)), rng.randint(0, num_class, 1000))


def save_sklearn_as_tl(model_dir, model):
    model = treelite.sklearn.import_model(model)
    model.serialize(os.path.join(model_dir, "checkpoint.tl"))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 5])
def test_sklearn_classification_model(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_sklearn_rf_classifier(num_features, num_class)
    base_name = "sklearn_rf_{}_class_{}".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    save_sklearn_as_tl(model_dir, model)
    rng = np.random.RandomState(1133)
    X = rng.random((100, num_features)).astype(np.float32)
    expected_shap_sum = (
        model.predict_proba(X) if num_class > 2 else model.predict_proba(X)[:, 1]
    )
    run_classification_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        model.predict_proba(X),
        expected_shap_sum,
    )


@lru_cache
def get_sklearn_rf_regressor(num_features):
    rng = np.random.RandomState(234)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=20)
    return model.fit(rng.random((1000, num_features)), rng.random(1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_sklearn_rf_regression_model(
    client, model_repo, instance_kind, use_experimental_optimizations
):
    num_features = 50
    model = get_sklearn_rf_regressor(num_features)
    base_name = "sklearn__rf_regressor"
    model_dir = get_model_directory(model_repo, base_name)
    save_sklearn_as_tl(model_dir, model)
    rng = np.random.RandomState(22345)
    X = rng.random((100, num_features)).astype(np.float32)
    run_regression_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
    )


def get_cuml_classifier(num_features, num_class):
    cuml = pytest.importorskip("cuml")
    rng = np.random.RandomState(134)
    model = cuml.ensemble.RandomForestClassifier(n_estimators=20)
    return model.fit(rng.random((1000, num_features)), rng.randint(0, num_class, 1000))


def save_cuml_as_tl(model_dir, model):
    tl_model = model.convert_to_treelite_model()
    tl_model.to_treelite_checkpoint(os.path.join(model_dir, "checkpoint.tl"))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 5])
def test_cuml_classification_model(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_cuml_classifier(num_features, num_class)
    base_name = "cuml{}_class_{}".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    save_cuml_as_tl(model_dir, model)
    rng = np.random.RandomState(7372)
    X = rng.random((100, num_features)).astype(np.float32)

    expected_proba = model.predict_proba(X)
    expected_shap_sum = expected_proba

    run_classification_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        expected_proba,
        expected_shap_sum,
    )


def get_cuml_regressor(num_features):
    cuml = pytest.importorskip("cuml")
    rng = np.random.RandomState(134)
    model = cuml.ensemble.RandomForestRegressor(n_estimators=20)
    return model.fit(rng.random((1000, num_features)), rng.random(1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_cuml_regression_model(
    client, model_repo, instance_kind, use_experimental_optimizations
):
    num_features = 50
    model = get_cuml_regressor(num_features)
    base_name = "cuml_regressor"
    model_dir = get_model_directory(model_repo, base_name)
    save_cuml_as_tl(model_dir, model)
    rng = np.random.RandomState(25)
    X = rng.random((100, num_features)).astype(np.float32)
    run_regression_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
    )


@lru_cache
def get_sklearn_gbm_classifier(num_features, num_class):
    rng = np.random.RandomState(1236)
    model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=20, init="zero")
    return model.fit(rng.random((1000, num_features)), rng.randint(0, num_class, 1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
@pytest.mark.parametrize("num_class", [2, 5])
def test_sklearn_gbm_classification_model(
    client, model_repo, instance_kind, num_class, use_experimental_optimizations
):
    num_features = 50
    model = get_sklearn_gbm_classifier(num_features, num_class)
    base_name = "sklearn_gbm_{}_class_{}".format(num_class, instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    save_sklearn_as_tl(model_dir, model)
    rng = np.random.RandomState(1133)
    X = rng.random((100, num_features)).astype(np.float32)
    run_classification_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_class,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
        model.predict_proba(X),
        model._raw_predict(X),
    )


@lru_cache
def get_sklearn_gbm_regressor(num_features):
    rng = np.random.RandomState(1162)
    model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=20, init="zero")
    return model.fit(rng.random((1000, num_features)), rng.random(1000))


@pytest.mark.parametrize("use_experimental_optimizations", [True, False])
@pytest.mark.parametrize("instance_kind", available_instance_types)
def test_sklearn_gbm_regression_model(
    client, model_repo, instance_kind, use_experimental_optimizations
):
    num_features = 50
    model = get_sklearn_gbm_regressor(num_features)
    base_name = "sklearn_gbm_regressor_{}".format(instance_kind)
    model_dir = get_model_directory(model_repo, base_name)
    save_sklearn_as_tl(model_dir, model)
    rng = np.random.RandomState(1133)
    X = rng.random((100, num_features)).astype(np.float32)
    run_regression_model(
        client,
        base_name,
        model_repo,
        "treelite_checkpoint",
        X,
        instance_kind,
        num_features,
        use_experimental_optimizations,
        model.predict(X),
    )
