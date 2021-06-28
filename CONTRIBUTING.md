# Contributing to the Triton FIL backend

## How to Contribute
You can help improve the Triton FIL backend in any of the following ways:
- Submitting a bug report, feature request or documentation issue
- Proposing and implementing a new feature
- Implementing a feature or bug-fix for an outstanding issue

### Bug reports
When submitting a bug report, please include a *minimum* *reproducible*
example. Ideally, this should be a snippet of code that other developers can
copy, paste, and immediately run to try to reproduce the error. Please:
- Do include import statements and any other code necessary to immediately run
  your example
- Avoid examples that require other developers to download models or data
  unless you cannot reproduce the problem with synthetically-generated data

### Code Contributions
To contribute code to this project, please follow these steps:
1. Find an issue to work on or submit an issue documenting the problem you
   would like to work on.
2. Comment on the issue saying that you plan to work on it.
3. Review the implementation details section below for information to help you
   make your changes in a way that is consistent with the rest of the codebase.
4. Code!
5. Create your pull request.
6. Wait for other developers to review your code and update your PR as needed.
7. Once a PR is approved, it will be merged into the main branch.

#### Signing Your Work
* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
  * Any contribution which contains commits that are not Signed-Off will not be accepted.
* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```
* Full text of the DCO:
  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```
  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Tests
Currently, contributions to the FIL backend are tested by running a series of
end-to-end checks on a set of example models. To run these tests locally, you
can invoke the `qa/run_tests.sh` helper script with environment variable
`LOCAL` set to `1`. If the `TRITON_IMAGE` environment variable is set, this
Docker image will be used for the Triton server container. Otherwise,
`triton_fil` will be used as the image tag. In order to run these tests, you
will need [RAPIDS
cuML](https://rapids.ai/start.html),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html),
and a version of
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html) compiled with
GPU support. These can all be installed via
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
using the provided `environment.yml` configuration file as shown below:

```bash
conda env update -f qa/environment.yml
conda activate triton_test
LOCAL=1 ./qa/run_tests.sh
```

For reporting issues, it may be useful to run tests with the environment
variable `SHOW_ENV` set to 1 as well. This will include additional details
about your environment which may be important to debugging a problem:

```bash
LOCAL=1 SHOW_ENV=1 ./qa/run_tests.sh
```

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
      that needs to be returned
   4. Inference is performed on the input data, and the results are copied to
      the output tensor
   5. The response containing the output is sent via
      `TRITONBACKEND_ResponseSend`
   6. `TRITONBACKEND_RequestRelease` is called on the original request to
      signal that the backend has finished processing the request.

Functions which are called as part of top-level steps in this cycle are
implemented in `api.cu`.

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
