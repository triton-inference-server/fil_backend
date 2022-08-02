import triton_python_backend_utils as pb_utils
import json
import asyncio

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

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
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "input__0")

            # List of awaitables containing inflight inference responses.
            inference_response_awaits = []
            for model_name in ['model_logistic-cpu', 'model_oversample-cpu', 'model_RFC-cpu']:
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

            # Get the OUTPUT0 from the "pytorch" model inference response
            logistic_output0_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[0], "output__0")

            # Get the OUTPUT1 from the "addsub" model inference response
            oversample_output1_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[1], "output__0")
            
            RFC_output2_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[2], "output__0")
            
            ensembled = (logistic_output0_tensor.as_numpy() + oversample_output1_tensor.as_numpy() + RFC_output2_tensor.as_numpy()) / 3

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
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')