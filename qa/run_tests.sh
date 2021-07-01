#!/bin/bash
set -e

LOCAL=${LOCAL:-0}
USE_GPU=${USE_GPU:-1}
TRITON_IMAGE="${TRITON_IMAGE:-triton_fil}"
if [ -z $SHOW_ENV ]
then
  [ $LOCAL -eq 1 ] && SHOW_ENV=0 || SHOW_ENV=1
fi

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

model_repo="${test_dir}/model_repository"

[ -d $log_dir ] || mkdir $log_dir

convert_to_cpu() {
  model_dir="${1%/}"
  cpu_model_dir="${model_dir}-cpu"
  cp -r "${model_dir}" "${cpu_model_dir}"

  config_file="${cpu_model_dir}/config.pbtxt"

  sed -i 's/KIND_GPU/KIND_CPU/g' "${config_file}"

  name_line="$(grep '^name:' "${config_file}")"
  name="${name_line%\"}"
  name="${name#*\"}"
  cpu_name="${name}-cpu"
  sed -i "s/name:\ \"${name}\"/name:\ \"${cpu_name}\"/g" "${config_file}"

  echo "${cpu_name}"
}

if [ $SHOW_ENV -eq 1 ]
then
  echo '-------------------------------------------------------'
  echo '----------------- Environment details -----------------'
  echo '-------------------------------------------------------'
  echo ''
  echo '---------------------- nvidia-smi ---------------------'
  nvidia-smi
  echo ''
  echo '------------------------ conda ------------------------'
  conda info
  conda list -q
  if [ $LOCAL -eq 0 ]
  then
    echo ''
    echo '--------------------- environment ---------------------'
    env
    echo ''
  fi
  echo '-------------------------------------------------------'
fi

models=()

echo 'Generating example models...'
if [ $USE_GPU -eq 1 ]
then
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name xgboost \
    --depth 11 \
    --trees 2000 \
    --classes 3 \
    --features 500) )
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name xgboost_json \
    --format xgboost_json \
    --depth 7 \
    --trees 500 \
    --features 500 \
    --predict_proba) )
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name lightgbm \
    --format lightgbm \
    --type lightgbm \
    --depth 3 \
    --trees 2000) )
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name regression \
    --depth 25 \
    --features 400 \
    --trees 10 \
    --task regression) )
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name sklearn \
    --type sklearn \
    --depth 3 \
    --trees 10 \
    --features 500) )
  "$script_dir/convert_sklearn" "$test_dir/model_repository/sklearn/1/model.pkl"
  models+=( $(python ${test_dir}/generate_example_model.py \
    --name cuml \
    --type cuml \
    --depth 3 \
    --trees 10 \
    --features 500 \
    --task regression) )
  "$script_dir/convert_cuml.py" "$test_dir/model_repository/cuml/1/model.pkl"
fi

echo 'Generating CPU-only gradient-boosted models...'
cpu_models=()
for i in ${!models[@]}
do
  cpu_models+=( "$(convert_to_cpu "${model_repository}/${models[$i]}")" )
done

echo 'Starting Triton server...'
if [ $LOCAL -eq 1 ]
then
  if [ $USE_GPU -eq 1 ]
  then
    DOCKER_FLAG='--gpus=all'
  else
    DOCKER_FLAG=''
  fi
  container=$(docker run -d $DOCKER_FLAG -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/qa/L0_e2e/model_repository/:/models ${TRITON_IMAGE} tritonserver --model-repository=/models)

  function cleanup {
    docker logs $container > $log_dir/local_container.log 2>&1
    docker rm -f $container > /dev/null
  }

  trap cleanup EXIT
else
  tritonserver --model-repository="${model_reposiitory}" > /logs/server.log 2>&1 &
fi

echo 'Testing example models...'
if [ $USE_GPU -eq 1 ]
then
  SHARED_MEM_FLAG=''
else
  SHARED_MEM_FLAG='--shared_mem None'
fi
for i in ${!models[@]}
do
  echo "Starting tests of model ${models[$i]}..."
  echo "Performance statistics for ${models[$i]}:"
  if [ $i -eq 1 ]  # Test HTTP at most once because it is slower
  then
    python ${test_dir}/test_model.py --protocol http --name ${models[$i]} $SHARED_MEM_FLAG
  else
    python ${test_dir}/test_model.py --protocol grpc --name ${models[$i]} $SHARED_MEM_FLAG
  fi
  echo "Model ${models[$i]} executed successfully"
done
