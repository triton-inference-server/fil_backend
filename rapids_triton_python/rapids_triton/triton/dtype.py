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

import numpy as np

DTYPE_NAMES = {
    np.dtype("bool").str: "BOOL",
    np.dtype("uint8").str: "UINT8",
    np.dtype("uint16").str: "UINT16",
    np.dtype("uint32").str: "UINT32",
    np.dtype("uint64").str: "UINT64",
    np.dtype("int8").str: "INT8",
    np.dtype("int16").str: "INT16",
    np.dtype("int32").str: "INT32",
    np.dtype("int64").str: "INT64",
    np.dtype("float16").str: "FP16",
    np.dtype("float32").str: "FP32",
    np.dtype("float64").str: "FP64",
}


def dtype_to_triton_name(dtype):
    dtype = np.dtype(dtype).str
    return DTYPE_NAMES.get(dtype, "BYTES")
