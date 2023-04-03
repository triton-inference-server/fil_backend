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
import numpy as np
import pytest
from rapids_triton import Client
import treelite
from generate_example_model import generate_config
import generate_example_model
from joblib import Memory


@pytest.fixture(scope="session")
def memory(pytestconfig):
    """Use for caching via joblib"""
    return Memory(pytestconfig.getoption("model_cache_dir"), verbose=1)


@pytest.fixture(scope="session")
def client():
    """A RAPIDS-Triton client for submitting inference requests"""
    client = Client()
    client.wait_for_server(120)
    return client


@pytest.fixture(scope="session")
def model_repo(pytestconfig):
    """The path to the model repository directory"""
    return pytestconfig.getoption("repo")

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


def save_sklearn_as_tl(model_dir, model):
    model = treelite.sklearn.import_model(model)
    model.serialize(os.path.join(model_dir, "checkpoint.tl"))


def save_cuml_as_tl(model_dir, model):
    tl_model = model.convert_to_treelite_model()
    tl_model.to_treelite_checkpoint(os.path.join(model_dir, "checkpoint.tl"))


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
    storage_type="AUTO",
):
    # probability output
    if expected_proba is not None:
        generate_config(
            model_name,
            model_repo,
            features=num_features,
            num_classes=num_class,
            predict_proba=True,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
            storage_type=storage_type,
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
        features=num_features,
        num_classes=num_class,
        predict_proba=False,
        instance_kind=instance_kind,
        model_format=model_format,
        use_experimental_optimizations=use_experimental_optimizations,
        storage_type=storage_type,
    )
    shared_mem = "cuda" if instance_kind == "KIND_GPU" else None
    result = predict(client, model_name, X, shared_mem=shared_mem)
    np.testing.assert_equal(result["output__0"], expected_class)

    # issue #351 cuml models don't work with threshold
    if num_class == 2 and "cuml" not in model_name:
        # threshold
        generate_config(
            model_name,
            model_repo,
            features=num_features,
            num_classes=num_class,
            predict_proba=False,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
            threshold=0.9,
            storage_type=storage_type,
        )
        result = predict(client, model_name, X)
        np.testing.assert_equal(
            result["output__0"], np.greater(expected_proba[:, 1], 0.9)
        )

    # Shap output
    if instance_kind == "KIND_GPU" and expected_shap_sum is not None:
        generate_config(
            model_name,
            model_repo,
            features=num_features,
            num_classes=num_class,
            predict_proba=False,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
            generate_shap=True,
            storage_type=storage_type,
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
    generate_config(
        model_name,
        model_repo,
        features=num_features,
        num_classes=1,
        predict_proba=False,
        task="regression",
        instance_kind=instance_kind,
        model_format=model_format,
        use_experimental_optimizations=use_experimental_optimizations,
        generate_shap=False,
    )
    result = predict(client, model_name, X)
    np.testing.assert_allclose(result["output__0"], expected, rtol=1e-3, atol=1e-3)

    # Shap output
    if instance_kind == "KIND_GPU":
        generate_config(
            model_name,
            model_repo,
            features=num_features,
            num_classes=1,
            task="regression",
            predict_proba=False,
            instance_kind=instance_kind,
            model_format=model_format,
            use_experimental_optimizations=use_experimental_optimizations,
            generate_shap=True,
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


@pytest.mark.parametrize(
    "use_experimental_optimizations",
    [True, False],
    ids=lambda x: "exper_optim:" + str(x),
)
@pytest.mark.parametrize(
    "instance_kind", available_instance_types, ids=lambda x: "instance:" + str(x)
)
class TestModels:
    pass


@pytest.mark.parametrize("num_class", [2, 10], ids=lambda x: "num_class:" + str(x))
class TestClassifiers(TestModels):
    @pytest.mark.parametrize(
        "use_json", [True, False], ids=lambda x: "use_json:" + str(x)
    )
    def test_xgb(
        self,
        client,
        memory,
        model_repo,
        instance_kind,
        num_class,
        use_experimental_optimizations,
        use_json,
    ):
        num_features = 500
        X, y = memory.cache(generate_example_model.generate_classification_data)(
            num_class, cols=num_features, cat_cols=2 if use_json else 0, add_nans=True
        )
        model = memory.cache(generate_example_model.train_xgboost_classifier)(
            X, y, depth=11, trees=2000
        )
        base_name = "xgboost_{}_class_{}".format(num_class, instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        model.save_model(
            os.path.join(model_dir, "xgboost.json" if use_json else "xgboost.model")
        )
        run_classification_model(
            client,
            base_name,
            model_repo,
            "xgboost_json" if use_json else "xgboost",
            X.to_numpy(dtype=np.float32) if "to_numpy" in dir(X) else X,
            instance_kind,
            num_class,
            num_features,
            use_experimental_optimizations,
            model.predict(X),
            model.predict_proba(X),
            model.predict(X, output_margin=True),
            storage_type="SPARSE",
        )

    def test_lgbm_classification_model(
        self,
        client,
        memory,
        model_repo,
        instance_kind,
        num_class,
        use_experimental_optimizations,
    ):
        num_features = 50
        X, y = memory.cache(generate_example_model.generate_classification_data)(
            num_class, cols=num_features, cat_cols=2, add_nans=True
        )
        model = memory.cache(generate_example_model.train_lightgbm_classifier)(
            X, y, depth=3, trees=2000
        )
        base_name = "lgbm_{}_class_{}".format(num_class, instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        model.booster_.save_model(os.path.join(model_dir, "model.txt"))
        run_classification_model(
            client,
            base_name,
            model_repo,
            "lightgbm",
            X.to_numpy(dtype=np.float32) if "to_numpy" in dir(X) else X,
            instance_kind,
            num_class,
            num_features,
            use_experimental_optimizations,
            model.predict(X),
            model.predict_proba(X),
            model.predict(X, raw_score=True),
        )

    def test_sklearn(
        self,
        client,
        memory,
        model_repo,
        instance_kind,
        num_class,
        use_experimental_optimizations,
    ):
        num_features = 500
        X, y = memory.cache(generate_example_model.generate_classification_data)(
            num_class, cols=num_features
        )
        model = memory.cache(generate_example_model.train_sklearn_classifier)(
            X, y, depth=10, trees=100
        )
        base_name = "sklearn_rf_{}_class_{}".format(num_class, instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        save_sklearn_as_tl(model_dir, model)
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

    def test_cuml(
        self,
        client,
        memory,
        model_repo,
        instance_kind,
        num_class,
        use_experimental_optimizations,
    ):
        num_features = 500
        X, y = memory.cache(generate_example_model.generate_classification_data)(
            num_class, cols=num_features
        )
        model = memory.cache(generate_example_model.train_cuml_classifier)(
            X, y, depth=10, trees=1000
        )
        base_name = "cuml{}_class_{}".format(num_class, instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        save_cuml_as_tl(model_dir, model)

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

    def test_sklearn_gbm(
        self,
        client,
        memory,
        model_repo,
        instance_kind,
        num_class,
        use_experimental_optimizations,
    ):
        num_features = 50
        X, y = memory.cache(generate_example_model.generate_classification_data)(
            num_class, cols=num_features
        )
        model = memory.cache(generate_example_model.train_sklearn_gbm_classifier)(
            X, y, depth=10, trees=100
        )
        base_name = "sklearn_gbm_{}_class_{}".format(num_class, instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        save_sklearn_as_tl(model_dir, model)
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
            model._raw_predict(np.array(X, order="C")),
        )


class TestRegressors(TestModels):
    def test_xgb(
        self, client, memory, model_repo, instance_kind, use_experimental_optimizations
    ):
        num_features = 500
        X, y = memory.cache(generate_example_model.generate_regression_data)(
            1000, cols=num_features
        )
        model = memory.cache(generate_example_model.train_xgboost_regressor)(
            X, y, depth=11, trees=2000
        )
        base_name = "xgboost_regression_{}".format(instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        model.save_model(os.path.join(model_dir, "xgboost.model"))
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

    def test_lgbm(
        self, client, memory, model_repo, instance_kind, use_experimental_optimizations
    ):
        num_features = 400
        X, y = memory.cache(generate_example_model.generate_regression_data)(
            1000, cols=num_features
        )
        model = memory.cache(generate_example_model.train_lightgbm_regressor)(
            X, y, depth=25, trees=2000
        )

        base_name = "lgbm{}_reg".format(instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        model.booster_.save_model(os.path.join(model_dir, "model.txt"))
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

    def test_sklearn(
        self, client, memory, model_repo, instance_kind, use_experimental_optimizations
    ):
        num_features = 50
        X, y = memory.cache(generate_example_model.generate_regression_data)(
            1000, cols=num_features
        )
        model = memory.cache(generate_example_model.train_sklearn_regressor)(
            X, y, depth=25, trees=100
        )
        base_name = "sklearn__rf_regressor"
        model_dir = get_model_directory(model_repo, base_name)
        save_sklearn_as_tl(model_dir, model)
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

    def test_cuml(
        self, client, memory, model_repo, instance_kind, use_experimental_optimizations
    ):
        num_features = 500
        X, y = memory.cache(generate_example_model.generate_regression_data)(
            1000, cols=num_features
        )
        model = memory.cache(generate_example_model.train_cuml_regressor)(
            X, y, depth=25, trees=100
        )
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

    def test_sklearn_gbm(
        self, client, memory, model_repo, instance_kind, use_experimental_optimizations
    ):
        num_features = 50
        X, y = memory.cache(generate_example_model.generate_regression_data)(
            1000, cols=num_features
        )
        model = memory.cache(generate_example_model.train_sklearn_gbm_regressor)(
            X, y, depth=25, trees=100
        )
        base_name = "sklearn_gbm_regressor_{}".format(instance_kind)
        model_dir = get_model_directory(model_repo, base_name)
        save_sklearn_as_tl(model_dir, model)
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
