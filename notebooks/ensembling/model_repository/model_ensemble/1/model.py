import asyncio
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "input__0")

            # List of awaitables containing inflight inference responses.
            inference_response_awaits = []
            for model_name in ['model_logistic', 'model_oversample', 'model_RFC']:
                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name=model_name,
                    requested_output_names=["output__0"],
                    inputs=[in_0])

                # Store the awaitable inside the array. We don't need
                # the inference response immediately so we do not `await`
                # here.
                inference_response_awaits.append(infer_request.async_exec())

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(
                *inference_response_awaits)

            for infer_response in inference_responses:
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_response.error().message())

            logistic_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[0], "output__0")
                        
            oversample_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[1], "output__0")
            
            RFC_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[2], "output__0")
            
            ensembled = (np._from_dlpack(logistic_tensor.to_dlpack()) + np._from_dlpack(oversample_tensor.to_dlpack()) + np._from_dlpack(RFC_tensor.to_dlpack())) / 3
            ensembled = ensembled.__dlpack__()
            ensembled_tensor = pb_utils.Tensor.from_dlpack("output__0", ensembled)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[ensembled_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print('Cleaning up...')
