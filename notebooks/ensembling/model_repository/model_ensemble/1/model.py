import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import json
import asyncio

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
            logistic_tensor = from_dlpack(logistic_tensor.to_dlpack())

            oversample_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[1], "output__0")
            oversample_tensor = from_dlpack(oversample_tensor.to_dlpack())
            
            RFC_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[2], "output__0")
            RFC_tensor = from_dlpack(RFC_tensor.to_dlpack())
            
            ensembled = (logistic_tensor.as_numpy() + oversample_tensor.as_numpy() + RFC_tensor.as_numpy()) / 3
            ensembled_tensor = pb_utils.Tensor("output__0", ensembled.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            #
            # Because the infer_response of the models contains the final
            # outputs with correct output names, we can just pass the list
            # of outputs to the InferenceResponse object.
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[ensembled_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print('Cleaning up...')