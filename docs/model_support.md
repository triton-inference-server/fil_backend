<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Model Support and Limitations
The FIL backend is designed to accelerate inference for **tree-based models**.
If the model you are trying to deploy is not tree-based, consider using one of
Triton's other backends.

## Frameworks
The FIL backend supports most XGBoost and LightGBM models using their native
serialization formats. The FIL backend also supports the following model types
from [Scikit-Learn and cuML](https://github.com/triton-inference-server/fil_backend/blob/main/docs/sklearn_and_cuml.md) using Treelite's checkpoint serialization format:

- GradientBoostingClassifier
- GradientBoostingRegressor
- IsolationForest
- RandomForestRegressor
- ExtraTreesClassifier
- ExtraTreesRegressor

In addition, the FIL backend can perform inference on tree models from any
framework if they are first exported to Treelite's checkpoint serialization
format.

## Serialization Formats
The FIL backend currently supports the following serialization formats:

- XGBoost JSON (Version < 1.7)
- XGBoost Binary
- LightGBM Text
- Treelite binary checkpoint

The FIL backend does **not** support direct ingestion of Pickle files. The
Pickled model must be converted to one of the above formats before it can be
used in Triton.

## Version Compatibility
Until version 3.0 of Treelite, Treelite offered no backward compatibility
for its checkpoint format even among minor releases. Therefore, the version
of Treelite used to save a checkpoint had to exactly match the version used in
the FIL backend. Starting with version 3.0, Treelite supports checkpoint
output from any version of Treelite starting with 2.7 until the next major
release.

XGBoost's JSON format also changes periodically between minor versions, and
older versions of Treelite used in the FIL backend may not support those
changes.

The compatibility matrix for Treelite and XGBoost with the FIL backend is
shown below:

| Triton Version | Supported Treelite Version(s) | Supported XGBoost JSON Version(s) |
| -------------- | ----------------------------- | --------------------------------- |
| 21.08          | 1.3.0                         | <1.6                              |
| 21.09-21.10    | 2.0.0                         | <1.6                              |
| 21.11-22.02    | 2.1.0                         | <1.6                              |
| 22.03-22.06    | 2.3.0                         | <1.6                              |
| 22.07          | 2.4.0                         | <1.7                              |
| 22.08+         | 2.4.0; >=3.0.0,<4.0.0         | <1.7                              |

## Limitations
The FIL backend currently does not support any multi-output regression models.

## Double-Precision Support
While the FIL backend can load double-precision models, it performs all
computations in single-precision mode. This can lead to slight differences in
model output for frameworks like LightGBM which natively use double precision.
Support for double-precision execution is planned for an upcoming release.
