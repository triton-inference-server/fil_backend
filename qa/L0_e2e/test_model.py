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

import numpy as np
import pytest

from rapids_triton import Client
from rapids_triton.testing import get_random_seed, arrays_close

TOTAL_SAMPLES = 8192

def valid_shm_modes():
    modes = [None]
    if os.environ.get('CPU_ONLY', 0) == 0:
        modes.append('cuda')
    return modes


@pytest.fixture(scope='session')
def client():
    client = Client()
    client.wait_for_server(60)
    return client

@pytest.fixture
def model_inputs():
    np.random.seed(get_random_seed())
    return {
        input_name:
        np.random.rand(TOTAL_SAMPLES, 1).astype('float32')
        for input_name in ('input__0',)
    }

@pytest.fixture
def model_output_sizes():
    return {'output__0': TOTAL_SAMPLES * np.dtype('float32').itemsize}

def get_ground_truth(inputs):
    return {'output__0': inputs['input__0']}


@pytest.mark.parametrize("model_name", ['identity'])
@pytest.mark.parametrize("shared_mem", valid_shm_modes())
def test_model(client, model_name, shared_mem, model_inputs, model_output_sizes):
    result = client.predict(
        model_name, model_inputs, model_output_sizes, shared_mem=shared_mem
    )
    ground_truth = get_ground_truth(model_inputs)

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            result[output_name],
            ground_truth[output_name],
            atol=1e-5,
            assert_close=True
        )
