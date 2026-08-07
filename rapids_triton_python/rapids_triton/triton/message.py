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


class TritonMessage:
    """Adapter to read output from both GRPC and HTTP responses"""

    def __init__(self, message):
        self.message = message

    def __getattr__(self, attr):
        try:
            return getattr(self.message, attr)
        except AttributeError:
            try:
                return self.message[attr]
            except Exception:  # Re-raise AttributeError
                pass
            raise
