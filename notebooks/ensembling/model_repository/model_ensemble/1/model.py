import triton_python_backend_utils as pb_utils
import json
import asyncio
from torch.utils.dlpack import from_dlpack

class TritonPythonModel:
    def initialize(self, args):


        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "output__0")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            self.output0_config['data_type'])

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
        output0_dtype = self.output0_dtype

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

            logistic_tensor = logistic_tensor.to_dlpack()
            oversample_tensor = oversample_tensor.to_dlpack()
            RFC_tensor = RFC_tensor.to_dlpack()

            class DLPack:
                def __init__(self, tensor):
                    self.tensor = tensor
                
                def __dlpack__(self):
                    return self.tensor
                
                def __dlpack_device__(self):
                    return self.tensor.memory_type_id

            logistic_tensor = DLPack(logistic_tensor)
            oversample_tensor = DLPack(oversample_tensor)
            RFC_tensor = DLPack(RFC_tensor)

            def pb_tensor_to_numpy(pb_tensor):
                if pb_tensor.is_cpu():
                    return pb_tensor.as_numpy()
                else:
                    pytorch_tensor = from_dlpack(pb_tensor)
                return pytorch_tensor.cpu().numpy()

            ensemble = (pb_tensor_to_numpy(logistic_tensor) + pb_tensor_to_numpy(oversample_tensor) + pb_tensor_to_numpy(RFC_tensor)) / 3
            ensembled_tensor = pb_utils.Tensor("output__0", ensemble)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[ensembled_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print('Cleaning up...')