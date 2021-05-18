#!/bin/bash
set -e

TRITON_IMAGE="${TRITON_IMAGE:-triton_fil}"

cd "$(git rev-parse --show-toplevel)"

[ -d qa/logs ] || mkdir qa/logs

models=()

echo 'Generating example models...'
models+=( $(python qa/L0_e2e/generate_example_model.py \
  --name xgboost \
  --depth 11 \
  --trees 2000 \
  --classes 3 \
  --features 500) )
models+=( $(python qa/L0_e2e/generate_example_model.py \
  --name xgboost_json \
  --format xgboost_json \
  --depth 7 \
  --trees 500 \
  --features 500 \
  --predict_proba) )
models+=( $(python qa/L0_e2e/generate_example_model.py \
  --name lightgbm \
  --format lightgbm \
  --type lightgbm \
  --depth 3 \
  --trees 2000) )
models+=( $(python qa/L0_e2e/generate_example_model.py \
  --name regression \
  --depth 25 \
  --features 400 \
  --trees 10 \
  --task regression) )

echo 'Starting Triton server...'
container=$(docker run -d --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/qa/L0_e2e/model_repository/:/models ${TRITON_IMAGE} tritonserver --model-repository=/models)

function cleanup {
  docker logs $container > qa/logs/local_container.log 2>&1
  docker rm -f $container > /dev/null
}

trap cleanup EXIT

echo 'Testing example models...'
for i in ${!models[@]}
do
  echo "Starting tests of model ${models[$i]}..."
  echo "Performance statistics for ${models[$i]}:"
  if [ $i -eq 1 ]  # Test HTTP at most once because it is slower
  then
    python qa/L0_e2e/test_model.py --protocol http --name ${models[$i]}
  else
    python qa/L0_e2e/test_model.py --protocol grpc --name ${models[$i]}
  fi
  echo "Model ${models[$i]} executed successfully"
done
