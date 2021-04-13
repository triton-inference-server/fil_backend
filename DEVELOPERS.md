# For Developers
<!--- TODO: Expand these notes --->

Given that Triton and cuML have fairly distinct coding styles and conventions,
the code for this backend is structured so as to separate Triton code and cuML
code as much as possible. `api.cu` contains the implementation of the backend
API expected by Triton and calls out to helper functions which help translate
across the cuML/Triton boundary. Looking at this file is probably the best
starting place for understanding the overall logic of this backend.
