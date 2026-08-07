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
from rapids_triton.client import Client
from rapids_triton.logging import logger
from rapids_triton.triton.client import STANDARD_PORTS


def arrays_close(
    a, b, atol=None, rtol=None, total_atol=None, total_rtol=None, assert_close=False
):
    """
    Compare numpy arrays for approximate equality

    :param numpy.array a: The array to compare against a reference value
    :param numpy.array b: The reference array to compare against
    :param float atol: The maximum absolute difference allowed between an
        element in a and an element in b before they are considered non-close.
        If both atol and rtol are set to None, atol is assumed to be 0. If atol
        is set to None and rtol is not None, no absolute threshold is used in
        comparisons.
    :param float rtol: The maximum relative difference allowed between an
        element in a and an element in b before they are considered non-close.
        If rtol is set to None, no relative threshold is used in comparisons.
    :param int total_atol: The maximum number of elements allowed to be
        non-close before the arrays are considered non-close.
    :param float total_rtol: The maximum proportion of elements allowed to be
        non-close before the arrays are considered non-close.
    """

    if np.any(a.shape != b.shape):
        if assert_close:
            raise AssertionError(
                "Arrays have different shapes:\n{} vs. {}".format(a.shape, b.shape)
            )
        return False

    if a.size == 0 and b.size == 0:
        return True

    if atol is None and rtol is None:
        atol = 0
    if total_atol is None and total_rtol is None:
        total_atol = 0

    diff_mask = np.ones(a.shape, dtype="bool")

    diff = np.abs(a - b)

    if atol is not None:
        diff_mask = np.logical_and(diff_mask, diff > atol)

    if rtol is not None:
        diff_mask = np.logical_and(diff_mask, diff > rtol * np.abs(b))

    is_close = True

    mismatch_count = np.sum(diff_mask)

    if total_atol is not None and mismatch_count > total_atol:
        is_close = False

    mismatch_proportion = mismatch_count / a.size
    if total_rtol is not None and mismatch_proportion > total_rtol:
        is_close = False

    if assert_close and not is_close:
        total_tol_desc = []
        if total_atol is not None:
            total_tol_desc.append(str(int(total_atol)))
        if total_rtol is not None:
            total_tol_desc.append("{:.2f} %".format(total_rtol * 100))
        total_tol_desc = " or ".join(total_tol_desc)

        msg = """Arrays have more than {} mismatched elements.

Mismatch in {} ({:.2f} %) elements
 a: {}
 b: {}

 Mismatched indices: {}""".format(
            total_tol_desc,
            mismatch_count,
            mismatch_proportion * 100,
            a,
            b,
            np.transpose(np.nonzero(diff_mask)),
        )
        raise AssertionError(msg)
    return is_close


def get_random_seed():
    """Provide random seed to allow for easier reproduction of testing failures

    Note: Code taken directly from cuML testing infrastructure"""
    current_random_seed = os.getenv("PYTEST_RANDOM_SEED")
    if current_random_seed is not None and current_random_seed.isdigit():
        random_seed = int(current_random_seed)
    else:
        random_seed = np.random.randint(0, 1e6)
        os.environ["PYTEST_RANDOM_SEED"] = str(random_seed)
    logger.info("Random seed value: %d", random_seed)
    return random_seed
