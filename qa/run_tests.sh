#!/bin/bash
set -e

LOCAL=${LOCAL:-0}
TRITON_IMAGE="${TRITON_IMAGE:-triton_fil}"

if [ $LOCAL -eq 1 ]
then
  cd "$(git rev-parse --show-toplevel)"
  log_dir=qa/logs
  test_dir=qa/L0_e2e
  script_dir=scripts
else
  log_dir=/logs
  test_dir=/triton_fil/qa/L0_e2e
  script_dir=/triton_fil/scripts
fi

[ -d $log_dir ] || mkdir $log_dir

models=()

echo 'Generating example models...'
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name xgboost \
#   --depth 11 \
#   --trees 2000 \
#   --classes 3 \
#   --features 500) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name xgboost_json \
#   --format xgboost_json \
#   --depth 7 \
#   --trees 500 \
#   --features 500 \
#   --predict_proba) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name lightgbm \
#   --format lightgbm \
#   --type lightgbm \
#   --depth 3 \
#   --trees 2000) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name regression \
#   --depth 25 \
#   --features 400 \
#   --trees 10 \
#   --task regression) )
models+=( $(python ${test_dir}/generate_example_model.py \
  --name xgboost-cpu \
  --instance_kind cpu \
  --depth 11 \
  --trees 2000 \
  --classes 3 \
  --features 500) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name xgboost_json-cpu \
#   --instance_kind cpu \
#   --format xgboost_json \
#   --depth 7 \
#   --trees 500 \
#   --features 500 \
#   --predict_proba) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name lightgbm-cpu \
#   --instance_kind cpu \
#   --format lightgbm \
#   --type lightgbm \
#   --depth 3 \
#   --trees 2000) )
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name regression-cpu \
#   --instance_kind cpu \
#   --depth 25 \
#   --features 400 \
#   --trees 10 \
#   --task regression) )
# 
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name sklearn \
#   --type sklearn \
#   --depth 3 \
#   --trees 10 \
#   --features 500) )
# "$script_dir/convert_sklearn" "$test_dir/model_repository/sklearn/1/model.pkl"
# models+=( $(python ${test_dir}/generate_example_model.py \
#   --name cuml \
#   --type cuml \
#   --depth 3 \
#   --trees 10 \
#   --features 500 \
#   --task regression) )
# "$script_dir/convert_cuml.py" "$test_dir/model_repository/cuml/1/model.pkl"

echo 'Starting Triton server...'
if [ $LOCAL -eq 1 ]
then
  container=$(docker run -d --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/qa/L0_e2e/model_repository/:/models ${TRITON_IMAGE} tritonserver --model-repository=/models)

  function cleanup {
    docker logs $container > $log_dir/local_container.log 2>&1
    docker rm -f $container > /dev/null
  }

  trap cleanup EXIT
else
  tritonserver --model-repository=${test_dir}/model_repository > /logs/server.log 2>&1 &
fi

echo 'Testing example models...'
for i in ${!models[@]}
do
  echo "Starting tests of model ${models[$i]}..."
  echo "Performance statistics for ${models[$i]}:"
  if [ $i -eq 1 ]  # Test HTTP at most once because it is slower
  then
    python ${test_dir}/test_model.py --protocol http --name ${models[$i]}
  else
    python ${test_dir}/test_model.py --protocol grpc --name ${models[$i]}
  fi
  echo "Model ${models[$i]} executed successfully"
done
