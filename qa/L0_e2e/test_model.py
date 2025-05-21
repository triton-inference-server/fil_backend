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
import pickle
from collections import defaultdict, namedtuple
from functools import lru_cache

try:
    import cuml
except Exception:
    cuml = None
import numpy as np
import pytest
import treelite
import xgboost as xgb
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from rapids_triton import Client
from rapids_triton.testing import arrays_close, get_random_seed

TOTAL_SAMPLES = 20
MODELS = (
    "xgboost",
    "xgboost_shap",
    "xgboost_json",
    "xgboost_ubj",
    "lightgbm",
    "lightgbm_rf",
    "regression",
    "sklearn",
    "cuml",
)

ModelData = namedtuple(
    "ModelData",
    (
        "name",
        "input_shapes",
        "output_sizes",
        "max_batch_size",
        "ground_truth_model",
        "config",
    ),
)


# TODO(wphicks): Replace with cache in 3.9
@lru_cache()
def valid_shm_modes():
    """Return a tuple of allowed shared memory modes"""
    modes = [None]
    if os.environ.get("CPU_ONLY", 0) == 0:
        modes.append("cuda")
    return tuple(modes)


# TODO(hcho3): Remove once we fix the flakiness of CUDA shared mem
# See https://github.com/triton-inference-server/server/issues/7688
def shared_mem_parametrize():
    params = [None]
    if "cuda" in valid_shm_modes():
        params.append(
            pytest.param("cuda", marks=pytest.mark.skip(reason="shared mem is flaky")),
        )
    return params


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


@pytest.fixture(scope="session")
def skip_shap(pytestconfig):
    return pytestconfig.getoption("no_shap")


def get_model_parameter(config, param, default=None):
    """Retrieve custom model parameters from config"""
    param_str = config.parameters[param].string_value
    if param_str:
        return param_str
    else:
        return default


class GTILModel:
    """A compatibility wrapper for executing models with GTIL"""

    def __init__(self, model_path, model_format, output_class):
        if model_format == "xgboost":
            self.tl_model = treelite.frontend.load_xgboost_model_legacy_binary(
                model_path
            )
        elif model_format == "xgboost_json":
            self.tl_model = treelite.frontend.load_xgboost_model(
                model_path, format_choice="json"
            )
        elif model_format == "xgboost_ubj":
            self.tl_model = treelite.frontend.load_xgboost_model(
                model_path, format_choice="ubjson"
            )
        elif model_format == "lightgbm":
            self.tl_model = treelite.frontend.load_lightgbm_model(model_path)
        elif model_format == "treelite_checkpoint":
            self.tl_model = treelite.Model.deserialize(model_path)
        self.output_class = output_class

    def _predict(self, arr):
        result = treelite.gtil.predict(self.tl_model, arr)
        # GTIL always returns prediction result with dimensions
        # (num_row, num_target, num_class)
        assert len(result.shape) == 3
        # We don't test multi-target models
        # TODO(hcho3): Add coverage for multi-target models
        assert result.shape[1] == 1
        return result[:, 0, :]

    def predict_proba(self, arr):
        result = self._predict(arr)
        if result.shape[1] > 1:
            return result
        else:
            return np.hstack((1 - result, result))

    def predict(self, arr):
        if self.output_class:
            return np.argmax(self.predict_proba(arr), axis=1)
        else:
            return self._predict(arr).squeeze()


class GroundTruthModel:
    """A reference model used for comparison against results returned from
    Triton"""

    def __init__(
        self,
        name,
        model_repo,
        model_format,
        predict_proba,
        output_class,
        use_cpu,
        *,
        model_version=1,
    ):
        model_dir = os.path.join(model_repo, name, f"{model_version}")
        self.predict_proba = predict_proba
        self._run_treeshap = False

        if model_format == "xgboost":
            model_path = os.path.join(model_dir, "xgboost.model")
        elif model_format == "xgboost_json":
            model_path = os.path.join(model_dir, "xgboost.json")
        elif model_format == "xgboost_ubj":
            model_path = os.path.join(model_dir, "xgboost.ubj")
        elif model_format == "lightgbm":
            model_path = os.path.join(model_dir, "model.txt")
        elif model_format == "treelite_checkpoint":
            if use_cpu:
                model_path = os.path.join(model_dir, "checkpoint.tl")
            else:
                model_path = os.path.join(model_dir, "model.pkl")
        else:
            raise RuntimeError("Model format not recognized")

        if name == "xgboost_shap":
            self._xgb_model = xgb.Booster()
            self._xgb_model.load_model(model_path)
            self._run_treeshap = True

        if use_cpu:
            self._base_model = GTILModel(model_path, model_format, output_class)
        else:
            if model_format == "treelite_checkpoint":
                with open(model_path, "rb") as pkl_file:
                    self._base_model = pickle.load(pkl_file)
            else:
                self._base_model = cuml.ForestInference.load(
                    model_path, output_class=output_class, model_type=model_format
                )

    def predict(self, inputs):
        if self.predict_proba:
            result = self._base_model.predict_proba(inputs["input__0"])
        else:
            result = self._base_model.predict(inputs["input__0"])
        output = {"output__0": result.squeeze()}
        if self._run_treeshap:
            treeshap_result = self._xgb_model.predict(
                xgb.DMatrix(inputs["input__0"]), pred_contribs=True
            )
            output["treeshap_output"] = treeshap_result
        return output


@pytest.fixture(scope="session", params=MODELS)
def model_data(request, client, model_repo):
    """All data associated with a model required for generating examples and
    comparing with ground truth results"""
    name = request.param
    config = client.get_model_config(name)
    input_shapes = {input_.name: list(input_.dims) for input_ in config.input}
    output_sizes = {
        output.name: np.prod(output.dims) * np.dtype("float32").itemsize
        for output in config.output
    }
    max_batch_size = config.max_batch_size

    model_format = get_model_parameter(config, "model_type", default="xgboost")
    predict_proba = get_model_parameter(config, "predict_proba", default="false")
    predict_proba = predict_proba == "true"
    output_class = get_model_parameter(config, "output_class", default="true")
    output_class = output_class == "true"

    use_cpu = config.instance_group[0].kind != 1

    ground_truth_model = GroundTruthModel(
        name,
        model_repo,
        model_format,
        predict_proba,
        output_class,
        use_cpu,
        model_version=1,
    )

    return ModelData(
        name, input_shapes, output_sizes, max_batch_size, ground_truth_model, config
    )


@pytest.mark.parametrize("shared_mem", shared_mem_parametrize())
@given(hypothesis_data=st.data())
@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much),
)
def test_small(shared_mem, client, model_data, hypothesis_data):
    """Test Triton-served model on many small Hypothesis-generated examples"""

    if model_data.name == "lightgbm":
        pytest.skip(
            reason=(
                "Legacy FIL gives incorrect output for latest LightGBM. "
                "See https://github.com/triton-inference-server/fil_backend/issues/432"
            )
        )

    all_model_inputs = defaultdict(list)
    total_output_sizes = {}
    all_triton_outputs = defaultdict(list)
    default_arrays = {
        name: np.random.rand(TOTAL_SAMPLES, *shape).astype("float32")
        for name, shape in model_data.input_shapes.items()
    }

    for i in range(TOTAL_SAMPLES):
        model_inputs = {
            name: hypothesis_data.draw(
                st.one_of(
                    st.just(default_arrays[name][i : i + 1, :]),
                    st_arrays("float32", [1] + shape),
                )
            )
            for name, shape in model_data.input_shapes.items()
        }
        if model_data.name == "sklearn" or model_data.name == "xgboost_shap":
            for array in model_inputs.values():
                assume(not np.any(np.isnan(array)))
        model_output_sizes = {
            name: size for name, size in model_data.output_sizes.items()
        }
        result = client.predict(
            model_data.name,
            model_inputs,
            model_data.output_sizes,
            shared_mem=shared_mem,
        )
        for name, input_ in model_inputs.items():
            all_model_inputs[name].append(input_)
        for name, size in model_output_sizes.items():
            total_output_sizes[name] = total_output_sizes.get(name, 0) + size
        for name, output in result.items():
            all_triton_outputs[name].append(output)

    all_model_inputs = {
        name: np.concatenate(arrays) for name, arrays in all_model_inputs.items()
    }
    all_triton_outputs = {
        name: np.concatenate(arrays) for name, arrays in all_triton_outputs.items()
    }

    try:
        ground_truth = model_data.ground_truth_model.predict(all_model_inputs)
    except Exception:
        assume(False)

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True,
            )
        else:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_atol=3,
                assert_close=True,
            )

    # Test entire batch of Hypothesis-generated inputs at once
    all_triton_outputs = client.predict(
        model_data.name,
        all_model_inputs,
        total_output_sizes,
        shared_mem=shared_mem,
    )

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True,
            )
        else:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_atol=3,
                assert_close=True,
            )


@pytest.mark.parametrize("shared_mem", shared_mem_parametrize())
def test_max_batch(client, model_data, shared_mem):
    """Test processing of a single maximum-sized batch"""
    max_inputs = {
        name: np.random.rand(model_data.max_batch_size, *shape).astype("float32")
        for name, shape in model_data.input_shapes.items()
    }
    model_output_sizes = {
        name: size * model_data.max_batch_size
        for name, size in model_data.output_sizes.items()
    }
    result = client.predict(
        model_data.name,
        max_inputs,
        model_output_sizes,
        shared_mem=shared_mem,
    )

    ground_truth = model_data.ground_truth_model.predict(max_inputs)

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba:
            arrays_close(
                result[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True,
            )
        else:
            arrays_close(
                result[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_rtol=3,
                assert_close=True,
            )
