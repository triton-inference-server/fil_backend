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
from collections import defaultdict
from functools import lru_cache

import cuml
import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from rapids_triton import Client
from rapids_triton.testing import get_random_seed, arrays_close

TOTAL_SAMPLES = 15
MODELS = (
    'xgboost',
    'xgboost_json',
    'lightgbm',
    'regression',
    'sklearn',
    'cuml'
)

# TODO(wphicks): Replace with cache in 3.9
@lru_cache()
def valid_shm_modes():
    modes = [None]
    if os.environ.get('CPU_ONLY', 0) == 0:
        modes.append('cuda')
    return tuple(modes)


@pytest.fixture(scope='session')
def client():
    client = Client()
    client.wait_for_server(120)
    return client


@pytest.fixture(scope='session')
def model_repo(pytestconfig):
    return pytestconfig.getoption('repo')


# TODO(wphicks): Turn these into fixtures
def get_input_shapes(client, model):
    try:
        return get_input_shapes.cache[model]
    except KeyError:
        config = client.get_model_config(model)
        get_input_shapes.cache[model] = {
            input_.name: list(input_.dims) for input_ in config.input
        }
    except AttributeError:
        get_input_shapes.cache = {}
    return get_input_shapes(client, model)


def get_output_sizes(client, model, dtype='float32'):
    try:
        return get_output_sizes.cache[model]
    except KeyError:
        config = client.get_model_config(model)
        get_output_sizes.cache[model] = {
            output.name: np.product(output.dims) * np.dtype(dtype).itemsize
            for output in config.output
        }
    except AttributeError:
        get_output_sizes.cache = {}
    return get_output_sizes(client, model)


def get_max_batch_size(client, model):
    try:
        return get_max_batch_size.cache[model]
    except KeyError:
        config = client.get_model_config(model)
        get_max_batch_size.cache[model] = config.max_batch_size
    except AttributeError:
        get_max_batch_size.cache = {}
    return get_max_batch_size(client, model)


def get_model_parameter(config, param, default=None):
    param_str = config.parameters[param].string_value
    if param_str:
        return param_str
    else:
        return default


class GroundTruthModel:
    model_cache = {}

    def __init__(
            self,
            name,
            model_repo,
            model_format,
            predict_proba,
            output_class,
            *,
            model_version=1):
        model_dir = os.path.join(model_repo, name, f'{model_version}')
        self.predict_proba = predict_proba

        if model_format == 'xgboost':
            model_path = os.path.join(model_dir, 'xgboost.model')
        elif model_format == 'xgboost_json':
            model_path = os.path.join(model_dir, 'xgboost.json')
        elif model_format == 'lightgbm':
            model_path = os.path.join(model_dir, 'model.txt')
        elif model_format == 'treelite_checkpoint':
            model_path = os.path.join(model_dir, 'model.pkl')
        else:
            raise RuntimeError('Model format not recognized')

        if model_format == 'treelite_checkpoint':
            with open(model_path, 'rb') as pkl_file:
                self._base_model = pickle.load(pkl_file)
        else:
            self._base_model = cuml.ForestInference.load(
                model_path, output_class=output_class, model_type=model_format
            )

    def predict(self, inputs):
        if self.predict_proba:
            result = self._base_model.predict_proba(inputs['input__0'])
        else:
            result = self._base_model.predict(inputs['input__0'])
        return {
            'output__0': result
        }

    @classmethod
    def get(cls, client, model, repo, *, version=1):
        model_key = (model, version)
        try:
            return cls.model_cache[model_key]
        except KeyError:
            config = client.get_model_config(model)
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

            cls.model_cache[model_key] = cls(
                model, repo, model_format, predict_proba, output_class,
                model_version=version
            )
        return cls.get(client, model, repo, version=version)


@pytest.mark.parametrize("model_name", MODELS)
@given(hypothesis_data=st.data())
@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much)
)
def test_model(client, model_repo, model_name, hypothesis_data):
    input_shapes = get_input_shapes(client, model_name)
    all_model_inputs = defaultdict(list)
    total_output_sizes = {}
    all_triton_outputs = defaultdict(list)
    default_arrays = {
        name: np.random.rand(TOTAL_SAMPLES, *shape).astype('float32')
        for name, shape in input_shapes.items()
    }

    for i in range(TOTAL_SAMPLES):
        model_inputs = {
            name: hypothesis_data.draw(
                st.one_of(
                    st.just(default_arrays[name][i:i+1, :]),
                    st_arrays('float32', [1] + shape)
                )
            ) for name, shape in input_shapes.items()
        }
        model_output_sizes = {
            name: size
            for name, size in get_output_sizes(client, model_name).items()
        }
        shared_mem = hypothesis_data.draw(st.one_of(
            st.just(mode) for mode in valid_shm_modes()
        ))
        result = client.predict(
            model_name, model_inputs, model_output_sizes, shared_mem=shared_mem
        )
        for name, input_ in model_inputs.items():
            all_model_inputs[name].append(input_)
        for name, size in model_output_sizes.items():
            total_output_sizes[name] = total_output_sizes.get(name, 0) + size
        for name, output in result.items():
            all_triton_outputs[name].append(output)

    gt_model = GroundTruthModel.get(client, model_name, model_repo)

    all_model_inputs = {
        name: np.concatenate(arrays)
        for name, arrays in all_model_inputs.items()
    }
    all_triton_outputs = {
        name: np.concatenate(arrays)
        for name, arrays in all_triton_outputs.items()
    }

    try:
        ground_truth = gt_model.predict(all_model_inputs)
    except Exception:
        assume(False)

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            all_triton_outputs[output_name],
            ground_truth[output_name],
            atol=1.5e-3,
            total_atol=2,
            assert_close=True
        )

    # Test entire batch of Hypothesis-generated inputs at once
    shared_mem = hypothesis_data.draw(st.one_of(
        st.just(mode) for mode in valid_shm_modes()
    ))
    all_triton_outputs = client.predict(
        model_name, all_model_inputs, total_output_sizes, shared_mem=shared_mem
    )

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            all_triton_outputs[output_name],
            ground_truth[output_name],
            atol=1.5e-3,
            total_atol=3,
            assert_close=True
        )


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("shared_mem", valid_shm_modes())
def test_max_batch(client, model_repo, model_name, shared_mem):
    input_shapes = get_input_shapes(client, model_name)
    gt_model = GroundTruthModel.get(client, model_name, model_repo)
    max_batch_size = get_max_batch_size(client, model_name)
    max_inputs = {
        name: np.random.rand(max_batch_size, *shape).astype('float32')
        for name, shape in input_shapes.items()
    }
    model_output_sizes = {
        name: size * max_batch_size
        for name, size in get_output_sizes(client, model_name).items()
    }
    shared_mem = valid_shm_modes()[0]
    result = client.predict(
        model_name, max_inputs, model_output_sizes, shared_mem=shared_mem
    )

    ground_truth = gt_model.predict(max_inputs)

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            result[output_name],
            ground_truth[output_name],
            atol=1.5e-3,
            total_atol=2,
            assert_close=True
        )
