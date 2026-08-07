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

from rapids_triton.triton.message import TritonMessage
from rapids_triton.utils.safe_import import ImportReplacement
from tritonclient import utils as triton_utils

try:
    import tritonclient.utils.cuda_shared_memory as shm
except OSError:  # CUDA libraries not available
    shm = ImportReplacement("tritonclient.utils.cuda_shared_memory")


def get_response_data(response, output_handle, output_name):
    """Convert Triton response to NumPy array"""
    if output_handle is None:
        return response.as_numpy(output_name)
    else:
        network_result = TritonMessage(response.get_output(output_name))
        return shm.get_contents_as_numpy(
            output_handle,
            triton_utils.triton_to_np_dtype(network_result.datatype),
            network_result.shape,
        )
