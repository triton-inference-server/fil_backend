#!/bin/bash
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

set -e

QA_DIR=$(cd $(dirname $0); pwd)
TEST_SCRIPT="$QA_DIR/run_tests.sh"

if [[ $TRITON_ENABLE_GPU != "OFF" ]]
then
  echo 'Running tests for GPU/CPU models...'
  MODEL_REPO="${QA_DIR}/L0_e2e/model_repository" "$TEST_SCRIPT"
else
  echo 'Running tests without visible GPUs...'
  CPU_ONLY=1 "$TEST_SCRIPT"
fi

