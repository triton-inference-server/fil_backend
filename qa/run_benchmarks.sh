#!/bin/bash
set -e

LOCAL=${LOCAL:-0}
TRITON_IMAGE="${TRITON_IMAGE:-triton_fil}"

if [ $LOCAL -eq 1 ]
then
  cd "$(git rev-parse --show-toplevel)"
  qa_dir=qa
  log_dir=qa/logs
  test_dir=qa/L0_e2e
  script_dir=scripts
else
  qa_dir=/triton_fil/qa
  log_dir=/logs
  test_dir=/triton_fil/qa/L0_e2e
  script_dir=/triton_fil/scripts
fi

model_repo="${test_dir}/model_repository"
cpu_model_repo="${model_repo}"

[ -d $log_dir ] || mkdir $log_dir
[ -d $model_repo ] || mkdir $model_repo

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

models=()

echo 'Generating benchmarking models...'
models+=( $(python ${test_dir}/generate_example_model.py \
  --name benchmark1 \
  --depth 8 \
  --trees 512 \
  --features 32 \
  --samples 2048 \
  --task regression) )
models+=( $(python ${test_dir}/generate_example_model.py \
  --name benchmark2 \
  --depth 16 \
  --trees 1024 \
  --features 256 \
  --samples 2048 \
  --task classification) )
models+=( $(python ${test_dir}/generate_example_model.py \
  --name benchmark3 \
  --depth 24 \
  --trees 2048 \
  --features 512 \
  --samples 2048 \
  --task classification) )

# for i in ${!models[@]}
# do
#   models+=( "$(convert_to_cpu "${models[$i]}")" )
# done

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

sleep 60

echo 'Running benchmarks...'
for i in ${!models[@]}
do
  echo "Benchmarking model ${models[$i]}..."
  "${qa_dir}/benchmark_model.sh" ${models[$i]} > "$log_dir/${models[$i]}.csv"
done
echo 'Benchmarks complete'
