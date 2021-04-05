# Contributing to the Triton FIL backend
<!--- TODO: Add basic contribution stuff --->

## Implementation Details

### Triton Classes
#### `ModelState`
Each model served by Triton is associated with a `ModelState` object. This
object stores any necessary information about the FIL model to be used.
Specifically, this includes the parameters read from `config.pbtxt` and a
handle to the model as parsed and loaded by Treelite.

#### `ModelInstanceState`
Triton can load up individual *instances* of a model to handle requests, and
any necessary data for these instance is stored in a `ModelInstanceState`
object. This includes an actual FIL `forest_t` object and a RAFT handle used
for handling GPU memory in a manner consistent with cuML.

### The Triton Request/Response Cycle
When an inference request is received by the Triton server, a series of
functions implemented in the backend code are executed at different points in
the request/response cycle. This process is briefly summarized as follows:
1. Server receives an inference request
2. If the backend associated with request is not yet loaded, load it and run
   [`TRITONBACKEND_Initialize`](https://github.com/wphicks/triton_fil_backend/blob/1205c57263e796512210b24bb0c04e3491564c74/src/api.cu#L55).
3. If the model has not yet been initialized, call
   [`TRITONBACKEND_ModelInitialize`](https://github.com/wphicks/triton_fil_backend/blob/1205c57263e796512210b24bb0c04e3491564c74/src/api.cu#L79),
   which creates the `ModelState` object.
4. Model instances are initialized by calling
   [`TRITONBACKEND_ModelInstanceInitialize`](https://github.com/wphicks/triton_fil_backend/blob/1205c57263e796512210b24bb0c04e3491564c74/src/api.cu#L122).
5. Call `TRITONBACKEND_ModelInstanceExecute`, which actually runs the model and
   returns a response. This includes a number of steps
   1. A response object is created using `TRITONBACKEND_ResponseNew`
   2. The input tensor is interpreted by using `TRITONBACKEND_InputProperties`
      to retrieve information about the shape and datatype of the input array
   3. `TRITONBACKEND_ResponseOutput` is called to construct each output tensor
      that needs to be resturned
   4. Inference is performed on the input data, and the results are copied to
      the output tensor
   5. The response containing the output is sent via
      `TRITONBACKEND_ResponseSend`
   6. `TRITONBACKEND_RequestRelease` is called on the original request to
      signal that the backend has finished processing the request.

Functions which are called as part of top-level steps in this cycle are
implemented in `api.cu`.

### Helper functions
Because the style and conventions used in Triton code differ significantly from
those used in cuML/FIL, a number of helper functions are provided to encourage
a clean separation between Triton and FIL components. While these are
documented in
[`triton_utils.h`](https://github.com/wphicks/triton_fil_backend/blob/1205c57263e796512210b24bb0c04e3491564c74/src/triton_fil/triton_utils.h)
and
[`triton_buffer.cuh`](https://github.com/wphicks/triton_fil_backend/blob/1205c57263e796512210b24bb0c04e3491564c74/src/triton_fil/triton_buffer.cuh),
a few of these are worth discussing in more detail here.

#### `get_input_buffers`
This helper function is used to construct the device buffers that can be passed
directly to a FIL model for inference. It determines whether Triton's input
data must be copied to a new device buffer and then wraps the pointer to device
memory in a `TritonBuffer` object which also keeps track of the shape of the
input array.

#### `get_output_buffers`
This helper function is used to construct the output buffers which will be used
to store output from the FIL model. These output buffers are similarly wrapped
in `TritonBuffer` objects

### Error Handling
In general, Triton signals that an unrecoverable error in inference has
occurred by returning a non-null `TRITONSERVER_Error` pointer. To avoid
excessive use of macros and redundant error checking around FIL code, the FIL
backend instead throws a `TritonException`, which wraps the
`TRITONSERVER_Error` pointer. If an unrecoverable error is encountered at any
point in the FIL backend code, this exception should be thrown. It will then be
caught in one of the Triton API functions implemented in `api.cu`, and the
appropriate `TRITONSERVER_Error` pointer will be returned. `api.cu` is the only
place in the FIL backend code where a `TRITONSERVER_Error` pointer should be
returned directly.

If a Triton function which returns such a pointer must be invoked as part of a
helper function, it should be wrapped in a `triton_check` call, which will
convert the pointer to an exception.
