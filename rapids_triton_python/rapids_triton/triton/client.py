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

import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http

STANDARD_PORTS = {"http": 8000, "grpc": 8001}


def get_triton_client(protocol="grpc", host="localhost", port=None, concurrency=4):
    """Get Triton client instance of desired type"""

    if port is None:
        port = STANDARD_PORTS[protocol]

    if protocol == "grpc":
        client = triton_grpc.InferenceServerClient(url=f"{host}:{port}", verbose=False)
    elif protocol == "http":
        client = triton_http.InferenceServerClient(
            url=f"{host}:{port}", verbose=False, concurrency=concurrency
        )
    else:
        raise RuntimeError('Bad protocol: "{}"'.format(protocol))

    return client
