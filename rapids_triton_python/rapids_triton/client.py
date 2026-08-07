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

import concurrent.futures
import time
from collections import namedtuple

from rapids_triton.triton.client import get_triton_client
from rapids_triton.triton.dtype import dtype_to_triton_name
from rapids_triton.triton.io import (
    create_triton_input,
    create_triton_output,
    destroy_shared_memory_region,
)
from rapids_triton.triton.response import get_response_data
from tritonclient import utils as triton_utils

# TODO(wphicks): Propagate device ids for cuda shared memory

MultiModelOutput = namedtuple("MultiModelOutput", ("name", "version", "output"))


class Client(object):
    def __init__(self, protocol="grpc", host="localhost", port=None, concurrency=4):
        self.triton_client = get_triton_client(
            protocol=protocol, host=host, port=port, concurrency=concurrency
        )
        self._protocol = protocol

    @property
    def protocol(self):
        return self._protocol

    def create_inputs(self, array_inputs, shared_mem=None):
        return [
            create_triton_input(
                self.triton_client,
                arr,
                name,
                dtype_to_triton_name(arr.dtype),
                protocol=self.protocol,
                shared_mem=shared_mem,
            )
            for name, arr in array_inputs.items()
        ]

    def create_outputs(self, output_sizes, shared_mem=None):
        return {
            name: create_triton_output(
                self.triton_client,
                size,
                name,
                protocol=self.protocol,
                shared_mem=shared_mem,
            )
            for name, size in output_sizes.items()
        }

    def wait_for_server(self, timeout):
        server_wait_start = time.time()
        while True:
            try:
                if self.triton_client.is_server_ready():
                    break
            except triton_utils.InferenceServerException:
                pass
            if time.time() - server_wait_start > timeout:
                raise RuntimeError("Server startup timeout expired")
            time.sleep(1)

    def clear_shared_memory(self):
        self.triton_client.unregister_cuda_shared_memory()
        self.triton_client.unregister_system_shared_memory()

    def release_io(self, io_objs):
        for io_ in io_objs:
            if io_.name is not None:
                self.triton_client.unregister_cuda_shared_memory(name=io_.name)
                destroy_shared_memory_region(io_.handle, shared_mem="cuda")

    def get_model_config(self, model_name):
        return self.triton_client.get_model_config(model_name).config

    def predict(
        self,
        model_name,
        input_data,
        output_sizes,
        model_version="1",
        shared_mem=None,
        attempts=1,
    ):
        model_version = str(model_version)

        try:
            inputs = self.create_inputs(input_data, shared_mem=shared_mem)
            outputs = self.create_outputs(output_sizes, shared_mem=shared_mem)

            response = self.triton_client.infer(
                model_name,
                model_version=model_version,
                inputs=[input_.input for input_ in inputs],
                outputs=[output_.output for output_ in outputs.values()],
            )
            result = {
                name: get_response_data(response, handle, name)
                for name, (_, handle, _) in outputs.items()
            }
            self.release_io(inputs)
            self.release_io(outputs.values())

        except triton_utils.InferenceServerException:
            if attempts > 1:
                return self.predict(
                    model_name,
                    input_data,
                    output_sizes,
                    model_version=model_version,
                    shared_mem=shared_mem,
                    attempts=attempts - 1,
                )
            raise
        return result

    def predict_async(
        self,
        model_name,
        input_data,
        output_sizes,
        model_version="1",
        shared_mem=None,
        attempts=1,
    ):
        model_version = str(model_version)

        inputs = self.create_inputs(input_data, shared_mem=shared_mem)
        outputs = self.create_outputs(output_sizes, shared_mem=shared_mem)

        future_result = concurrent.futures.Future()

        def callback(result, error):
            if error is None:
                output_arrays = {
                    name: get_response_data(result, handle, name)
                    for name, (_, handle, _) in outputs.items()
                }

                future_result.set_result(output_arrays)

                self.release_io(outputs.values())
            else:
                if isinstance(error, triton_utils.InferenceServerException):
                    if attempts > 1:
                        future_result.set_result(
                            self.predict(
                                model_name,
                                input_data,
                                output_sizes,
                                model_version=model_version,
                                shared_mem=shared_mem,
                                attempts=attempts - 1,
                            )
                        )
                future_result.set_exception(error)

        self.triton_client.async_infer(
            model_name,
            model_version=model_version,
            inputs=[input_.input for input_ in inputs],
            outputs=[output_.output for output_ in outputs.values()],
            callback=callback,
        )

        if shared_mem is not None:

            def release_callback(fut):
                self.release_io(inputs)

            future_result.add_done_callback(release_callback)
        return future_result

    def predict_multimodel_async(
        self,
        model_names,
        input_data,
        output_sizes,
        model_versions=("1",),
        shared_mem=None,
        executor=None,
        attempts=1,
    ):
        all_models = [
            (name, str(version)) for name in model_names for version in model_versions
        ]

        inputs = self.create_inputs(input_data, shared_mem=shared_mem)

        all_future_results = []
        for model_name, version in all_models:
            outputs = self.create_outputs(output_sizes, shared_mem=shared_mem)

            def create_callback(future_result, outputs):
                def callback(result, error):
                    if error is None:
                        output_arrays = {
                            name: get_response_data(result, handle, name)
                            for name, (_, handle, _) in outputs.items()
                        }

                        future_result.set_result(
                            MultiModelOutput(
                                name=model_name, version=version, output=output_arrays
                            )
                        )

                        self.release_io(outputs.values())
                    else:
                        if isinstance(error, triton_utils.InferenceServerException):
                            if attempts > 1:
                                future_result.set_result(
                                    self.predict(
                                        model_name,
                                        input_data,
                                        output_sizes,
                                        model_version=version,
                                        shared_mem=shared_mem,
                                        attempts=attempts - 1,
                                    )
                                )
                        future_result.set_exception(error)

                return callback

            all_future_results.append(concurrent.futures.Future())
            self.triton_client.async_infer(
                model_name,
                model_version=version,
                inputs=[input_.input for input_ in inputs],
                outputs=[output_.output for output_ in outputs.values()],
                callback=create_callback(all_future_results[-1], outputs),
            )

        def wait_for_all(future_results, releasable_inputs):
            concurrent.futures.wait(future_results)
            self.release_io(releasable_inputs)
            return [fut.result() for fut in future_results]

        if executor is None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(wait_for_all, all_future_results, inputs)
        else:
            return executor.submit(wait_for_all, all_future_results, inputs)
