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

RETRAIN=${RETRAIN:-0}

QA_DIR=$(cd $(dirname $0); pwd)
MODEL_REPO="${QA_DIR}/L0_e2e/model_repository"
CPU_MODEL_REPO="${QA_DIR}/L0_e2e/cpu_model_repository"

SCRIPTS_DIR="${QA_DIR}/../scripts"
MODEL_REPO="${QA_DIR}/L0_e2e/model_repository"
GENERATOR_SCRIPT="python ${QA_DIR}/L0_e2e/generate_example_model.py"

SKLEARN_CONVERTER="${SCRIPTS_DIR}/convert_sklearn"
CUML_CONVERTER="${SCRIPTS_DIR}/convert_cuml.py"

models=()

name=xgboost
if [ $RETRAIN -ne 0 ] || [ ! -d "${MODEL_REPO}/${name}" ]
then
  ${GENERATOR_SCRIPT} \
    --name $name \
    --depth 11 \
    --trees 2000 \
    --classes 3 \
    --features 500 \
    --storage_type SPARSE
  models+=( $name )
fi

name=xgboost_json
if [ $RETRAIN -ne 0 ] || [ ! -d "${MODEL_REPO}/${name}" ]
then
  ${GENERATOR_SCRIPT} \
    --name $name \
    --format xgboost_json \
    --depth 7 \
    --trees 500 \
    --features 500 \
    --predict_proba
  models+=( $name )
fi

mkdir -p "${CPU_MODEL_REPO}"
cp -r "${MODEL_REPO}"/* "${CPU_MODEL_REPO}"/

if [ ! -z $OWNER_ID ] && [ ! -z $OWNER_GID ]
then
  chown -R "${OWNER_ID}:${OWNER_GID}" "${MODEL_REPO}"
  chown -R "${OWNER_ID}:${OWNER_GID}" "${CPU_MODEL_REPO}"
fi

find "${CPU_MODEL_REPO}" -name 'config.pbtxt' -exec \
  sed -i s/KIND_GPU/KIND_CPU/g {} +
