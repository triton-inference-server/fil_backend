# Copyright (c) 2022, NVIDIA CORPORATION.
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


class ImportUnavailableError(Exception):
    """Error thrown if a symbol is unavailable due to an issue importing it"""


class ImportReplacement:
    """A class to be used in place of an importable symbol if that symbol
    cannot be imported

    Parameters
    ----------
    symbol: str
        The name or import path to be used in error messages when attempting to
        make use of this symbol. E.g. "some_pkg.func" would result in an
        exception with message "some_pkg.func could not be imported"
    """

    def __init__(self, symbol):
        self._msg = f"{symbol} could not be imported"

    def __getattr__(self, name):
        raise ImportUnavailableError(self._msg)

    def __call__(self, *args, **kwargs):
        raise ImportUnavailableError(self._msg)
