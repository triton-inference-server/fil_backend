#!/bin/bash
set -e

LOCAL=${LOCAL:-0}
TRITON_IMAGE="${TRITON_IMAGE:-triton_fil}"
if [ -z $SHOW_ENV ]
then
  [ $LOCAL -eq 1 ] && SHOW_ENV=0 || SHOW_ENV=1
fi

if [ $LOCAL -eq 1 ]
then
  cd "$(git rev-parse --show-toplevel)"
  log_dir=$PWD/qa/logs
  test_dir=$PWD/qa/L0_e2e
  script_dir=$PWD/scripts
  if [ -z $SERVER_GRACE ]
  then
    SERVER_GRACE=60
  fi
else
  log_dir=/logs
  test_dir=/triton_fil/qa/L0_e2e
  script_dir=/triton_fil/scripts
  if [ -z $SERVER_GRACE ]
  then
    SERVER_GRACE=180
  fi
fi

[ -d $log_dir ] || mkdir $log_dir

model_repo="${test_dir}/model_repository"
cpu_model_repo="${test_dir}/cpu_model_repository"

[ -d $model_repo ] || mkdir $model_repo
[ -d $cpu_model_repo ] || mkdir $cpu_model_repo

convert_to_cpu() {
  model_dir="${1}"
  cpu_model_dir="${cpu_model_repo}/${model_dir}-cpu"
  [ ! -d "$cpu_model_dir" ] || rm -r "$cpu_model_dir"
  cp -r "${model_repo}/${model_dir}" "${cpu_model_dir}"

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

echo 'Generating gradient-boosted test models...'
models+=( $(python ${test_dir}/generate_example_model.py \
  --name xgboost \
  --depth 11 \
  --trees 2000 \
  --classes 3 \
  --features 500 \
  --storage_type SPARSE) )
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
  --cat_features 3 \
  --depth 3 \
  --trees 2000) )
models+=( $(python ${test_dir}/generate_example_model.py \
  --name regression \
  --format lightgbm \
  --type lightgbm \
  --depth 25 \
  --features 400 \
  --trees 10 \
  --task regression) )

echo 'Generating CPU-only gradient-boosted models...'
cpu_models=()
for i in ${!models[@]}
do
  cpu_models+=( "$(convert_to_cpu "${models[$i]}")" )
done

echo 'Generating random forest test models...'
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
  --max_batch_size 32768 \
  --features 500 \
  --task regression) )
"$script_dir/convert_cuml.py" "$test_dir/model_repository/cuml/1/model.pkl"

function cleanup {
  if [ ! -z $container ]
  then
    docker logs $container > $log_file 2>&1 || true
    docker rm -f $container > /dev/null
  fi
  if [ ! -z $server_pid ]
  then
    count=0
    while kill -0 $server_pid >/dev/null 2>&1
    do
      if [ $count -lt 20 ]
      then
        kill -15 $server_pid
      else
        kill -9 $server_pid
      fi

      if [ $count -gt 30 ]
      then
        echo 'ERROR: Server could not be shut down!'
        kill -9 $server_pid
        break
      fi

      ((count=count+1))
      sleep 1
    done
  fi
}

trap cleanup EXIT

echo 'Starting Triton server for GPU models...'
log_file="$log_dir/gpu_tests.log"
if [ $LOCAL -eq 1 ]
then
  container=$(docker run -d --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$model_repo:/models" ${TRITON_IMAGE} tritonserver --model-repository=/models)
else
  tritonserver --model-repository="${model_repo}" > /logs/server.log 2>&1 &
  server_pid=$!
fi

echo 'Testing GPU models...'
for i in ${!models[@]}
do
  echo "Starting tests of model ${models[$i]}..."
  echo "Performance statistics for ${models[$i]}:"
  if [ $i -eq 1 ]  # Test HTTP at most once because it is slower
  then
    python ${test_dir}/test_model.py --server_grace $SERVER_GRACE --protocol http --name ${models[$i]}
  elif [ ${models[$i]} = 'cuml' ]  # Test large inputs for just one model
  then
    python ${test_dir}/test_model.py --server_grace $SERVER_GRACE --protocol grpc --name ${models[$i]} -b 32768
  else
    python ${test_dir}/test_model.py --server_grace $SERVER_GRACE --protocol grpc --name ${models[$i]}
  fi
  echo "Model ${models[$i]} executed successfully"
done

echo 'Testing statistics reporting...'
if perf_analyzer -m "${models[0]}" | grep -q 'Inference count: [0-9]*[1-9][0-9]*'
then
  echo 'Statistics reported successfully.'
else
  echo 'Failure in statistics reporting!'
  exit 1
fi

echo 'Starting Triton server for CPU models...'
cleanup
log_file="$log_dir/cpu_tests.log"
if [ $LOCAL -eq 1 ]
then
  container=$(docker run -d --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$cpu_model_repo:/models" ${TRITON_IMAGE} tritonserver --model-repository=/models)
else
  tritonserver --model-repository="${cpu_model_repo}" > /logs/server.log 2>&1 &
  server_pid=$!
fi

echo 'Testing CPU models...'
for i in ${!cpu_models[@]}
do
  echo "Starting tests of model ${cpu_models[$i]}..."
  echo "Performance statistics for ${cpu_models[$i]}:"
  python ${test_dir}/test_model.py --server_grace $SERVER_GRACE --protocol grpc --name ${cpu_models[$i]} --repo "${cpu_model_repo}"
  echo "Model ${cpu_models[$i]} executed successfully"
done

echo 'Starting Triton server without GPU...'
cleanup
log_file="$log_dir/no_gpu_tests.log"
if [ $LOCAL -eq 1 ]
then
  container=$(docker run -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$cpu_model_repo:/models" ${TRITON_IMAGE} tritonserver --model-repository=/models)
else
  CUDA_VISIBLE_DEVICES="" tritonserver --model-repository="${cpu_model_repo}" > /logs/server.log 2>&1 &
  server_pid=$!
fi

echo 'Testing CPU models without visible GPU...'
echo "Starting tests of model ${cpu_models[0]} without visible GPU..."
echo "Performance statistics for ${cpu_models[0]}:"
python ${test_dir}/test_model.py --shared_mem None --server_grace $SERVER_GRACE --protocol grpc --name ${cpu_models[0]} --repo "${cpu_model_repo}"
echo "Model ${cpu_models[$i]} executed successfully"
