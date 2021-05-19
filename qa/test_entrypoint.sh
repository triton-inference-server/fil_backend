#!/bin/bash
set -e

models=()

python -c 'import sys; print(sys.executable)'
python -c 'import tritonclient.http'

echo 'Generating example models...'
models+=( $(python /L0_e2e/generate_example_model.py \
  --name xgboost \
  --depth 11 \
  --trees 2000 \
  --classes 3 \
  --features 500) )
models+=( $(python /L0_e2e/generate_example_model.py \
  --name xgboost_json \
  --format xgboost_json \
  --depth 7 \
  --trees 500 \
  --features 500 \
  --predict_proba) )
models+=( $(python /L0_e2e/generate_example_model.py \
  --name lightgbm \
  --format lightgbm \
  --type lightgbm \
  --depth 3 \
  --trees 2000) )
models+=( $(python /L0_e2e/generate_example_model.py \
  --name regression \
  --depth 25 \
  --features 400 \
  --trees 10 \
  --task regression) )

echo 'Starting Triton server...'
tritonserver --model-repository=/L0_e2e/model_repository > /logs/server.log 2>&1 &

echo 'Testing example models...'
for i in ${!models[@]}
do
  echo "Starting tests of model ${models[$i]}..."
  echo "Performance statistics for ${models[$i]}:"
  if [ $i -eq 1 ]  # Test HTTP at most once because it is slower
  then
    python /L0_e2e/test_model.py --protocol http --name ${models[$i]}
  else
    python /L0_e2e/test_model.py --protocol grpc --name ${models[$i]}
  fi
  echo "Model ${models[$i]} executed successfully"
done
