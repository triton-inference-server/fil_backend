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
from hypothesis import given, settings, assume, HealthCheck, note
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from rapids_triton import Client
from rapids_triton.testing import get_random_seed, arrays_close
import xgboost as xgb

TOTAL_SAMPLES = 20
MODELS = (
    'xgboost',
    'xgboost_json',
    'lightgbm',
    'lightgbm_rf',
    'regression',
    'sklearn',
    'cuml'
)

ModelData = namedtuple('ModelData', (
    'name',
    'input_shapes',
    'output_sizes',
    'max_batch_size',
    'ground_truth_model',
    'config'
))

# TODO(wphicks): Replace with cache in 3.9
@lru_cache()
def valid_shm_modes():
    """Return a tuple of allowed shared memory modes"""
    modes = [None]
    if os.environ.get('CPU_ONLY', 0) == 0:
        modes.append('cuda')
    return tuple(modes)


@pytest.fixture(scope='session')
def client():
    """A RAPIDS-Triton client for submitting inference requests"""
    client = Client()
    client.wait_for_server(120)
    return client


@pytest.fixture(scope='session')
def model_repo(pytestconfig):
    """The path to the model repository directory"""
    return pytestconfig.getoption('repo')


@pytest.fixture(scope='session')
def skip_shap(pytestconfig):
    return pytestconfig.getoption('no_shap')


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
        if model_format == 'treelite_checkpoint':
            self.tl_model = treelite.Model.deserialize(model_path)
        else:
            self.tl_model = treelite.Model.load(model_path, model_format)
        self.output_class = output_class

    def _predict(self, arr):
        return treelite.gtil.predict(self.tl_model, arr)

    def predict_proba(self, arr):
        result = self._predict(arr)
        if len(result.shape) > 1:
            return result
        else:
            return np.transpose(np.vstack((1 - result, result)))

    def predict(self, arr):
        if self.output_class:
            return np.argmax(self.predict_proba(arr), axis=1)
        else:
            return self._predict(arr)


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
            model_version=1):
        model_dir = os.path.join(model_repo, name, f'{model_version}')
        self.predict_proba = predict_proba
        self.output_class= output_class
        self.use_cpu = use_cpu


        if use_cpu:
            self._run_treeshap = False
        else:
            self._run_treeshap = True

        if model_format == 'xgboost':
            model_path = os.path.join(model_dir, 'xgboost.model')
        elif model_format == 'xgboost_json':
            model_path = os.path.join(model_dir, 'xgboost.json')
        elif model_format == 'lightgbm':
            model_path = os.path.join(model_dir, 'model.txt')
        elif model_format == 'treelite_checkpoint':
            if use_cpu:
                model_path = os.path.join(model_dir, 'checkpoint.tl')
            else:
                model_path = os.path.join(model_dir, 'model.pkl')
        else:
            raise RuntimeError('Model format not recognized')

        if use_cpu:
            self._base_model = GTILModel(
                model_path, model_format, output_class
            )
            self._treelite_model = self._base_model.tl_model
        else:
            if model_format == 'treelite_checkpoint':
                with open(model_path, 'rb') as pkl_file:
                    self._base_model = pickle.load(pkl_file)
                self._treelite_model = self._base_model
            else:
                self._base_model = cuml.ForestInference.load(
                    model_path, output_class=output_class, model_type=model_format
                )
                self._treelite_model =  GTILModel(
                        model_path, model_format, output_class
                    ).tl_model

    def predict(self, inputs):
        if self.predict_proba:
            result = self._base_model.predict_proba(inputs['input__0'])
        else:
            result = self._base_model.predict(inputs['input__0'])
        output = {'output__0' : result}
        if self._run_treeshap:
            explainer = cuml.explainer.TreeExplainer(model=self._treelite_model)
            treeshap_result = explainer.shap_values(inputs['input__0'])
            # reshape to the same output as triton
            # append expected value as the last column
            if len(treeshap_result.shape) >= 3:
                treeshap_result = np.swapaxes(treeshap_result, 0, 1)
                treeshap_result = np.pad(treeshap_result, ((0,0),(0,0),(0,1)))
                for i in range(len(explainer.expected_value)):
                    treeshap_result[:,i,-1] = explainer.expected_value[i]
            else:
                treeshap_result = np.pad(treeshap_result, ((0,0),(0,1)))
                treeshap_result[:,-1] = explainer.expected_value

            output['treeshap_output'] = treeshap_result
        return output


@pytest.fixture(scope='session', params=MODELS)
def model_data(request, client, model_repo, skip_shap):
    """All data associated with a model required for generating examples and
    comparing with ground truth results"""
    name = request.param
    if skip_shap and name == 'xgboost_shap':
        pytest.skip("GPU Treeshap tests not enabled")
    config = client.get_model_config(name)
    input_shapes = {
        input_.name: list(input_.dims) for input_ in config.input
    }
    output_sizes = {
        output.name: np.product(output.dims) * np.dtype('float32').itemsize
        for output in config.output
    }
    max_batch_size = config.max_batch_size

    model_format = get_model_parameter(
        config, 'model_type', default='xgboost'
    )
    predict_proba = get_model_parameter(
        config, 'predict_proba', default='false'
    )
    predict_proba = (predict_proba == 'true')
    output_class = get_model_parameter(
        config, 'output_class', default='true'
    )
    output_class = (output_class == 'true')

    use_cpu = (config.instance_group[0].kind != 1)

    ground_truth_model = GroundTruthModel(
        name, model_repo, model_format, predict_proba, output_class, use_cpu,
        model_version=1
    )

    return ModelData(
        name,
        input_shapes,
        output_sizes,
        max_batch_size,
        ground_truth_model,
        config
    )


@given(hypothesis_data=st.data())
@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much)
)
def test_small(client, model_data, hypothesis_data):
    """Test Triton-served model on many small Hypothesis-generated examples"""
    all_model_inputs = defaultdict(list)
    total_output_sizes = {}
    all_triton_outputs = defaultdict(list)
    default_arrays = {
        name: np.random.rand(TOTAL_SAMPLES, *shape).astype('float32')
        for name, shape in model_data.input_shapes.items()
    }

    for i in range(TOTAL_SAMPLES):
        model_inputs = {
            name: hypothesis_data.draw(
                st.one_of(
                    st.just(default_arrays[name][i:i+1, :]),
                    st_arrays('float32', [1] + shape)
                )
            ) for name, shape in model_data.input_shapes.items()
        }
        
        if model_data.name == 'sklearn' or model_data.name == 'xgboost_shap':
            for array in model_inputs.values():
                assume(not np.any(np.isnan(array)))
        model_output_sizes = {
            name: size
            for name, size in model_data.output_sizes.items()
        }
        shared_mem = hypothesis_data.draw(st.one_of(
            st.just(mode) for mode in valid_shm_modes()
        ))
        result = client.predict(
            model_data.name, model_inputs, model_data.output_sizes,
            shared_mem=shared_mem
        )
        for name, input_ in model_inputs.items():
            all_model_inputs[name].append(input_)
        for name, size in model_output_sizes.items():
            total_output_sizes[name] = total_output_sizes.get(name, 0) + size
        for name, output in result.items():
            all_triton_outputs[name].append(output)

    all_model_inputs = {
        name: np.concatenate(arrays)
        for name, arrays in all_model_inputs.items()
    }
    all_triton_outputs = {
        name: np.concatenate(arrays)
        for name, arrays in all_triton_outputs.items()
    }

    ground_truth = model_data.ground_truth_model.predict(all_model_inputs)

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba and not "shap" in output_name:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True
            )
        else:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_atol=3,
                assert_close=True
            )
        
    # Test shapley values efficiency property
    if not model_data.ground_truth_model.predict_proba and not model_data.ground_truth_model.output_class:
        note(all_triton_outputs["treeshap_output"].sum(axis=-1))
        note(all_triton_outputs["output__0"])
        note(all_triton_outputs["output__0"] - all_triton_outputs["treeshap_output"].sum(axis=-1))
        arrays_close(all_triton_outputs["treeshap_output"].sum(axis=-1), all_triton_outputs["output__0"],atol=0.1,total_atol=3,
 assert_close=True)
                

    # Test entire batch of Hypothesis-generated inputs at once
    shared_mem = hypothesis_data.draw(st.one_of(
        st.just(mode) for mode in valid_shm_modes()
    ))
    all_triton_outputs = client.predict(
        model_data.name, all_model_inputs, total_output_sizes,
        shared_mem=shared_mem
    )

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True
            )
        else:
            arrays_close(
                all_triton_outputs[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_atol=3,
                assert_close=True
            )


@pytest.mark.parametrize("shared_mem", valid_shm_modes())
def test_max_batch(client, model_data, shared_mem):
    return
    """Test processing of a single maximum-sized batch"""
    max_inputs = {
        name: np.random.rand(model_data.max_batch_size, *shape).astype('float32')
        for name, shape in model_data.input_shapes.items()
    }
    model_output_sizes = {
        name: size * model_data.max_batch_size
        for name, size in model_data.output_sizes.items()
    }
    shared_mem = valid_shm_modes()[0]
    result = client.predict(
        model_data.name, max_inputs, model_output_sizes, shared_mem=shared_mem
    )

    ground_truth = model_data.ground_truth_model.predict(max_inputs)

    for output_name in sorted(ground_truth.keys()):
        if model_data.ground_truth_model.predict_proba:
            arrays_close(
                result[output_name],
                ground_truth[output_name],
                rtol=1e-3,
                atol=1e-2,
                assert_close=True
            )
        else:
            arrays_close(
                result[output_name],
                ground_truth[output_name],
                atol=0.1,
                total_rtol=3,
                assert_close=True
            )
