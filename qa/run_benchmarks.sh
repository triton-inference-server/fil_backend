#!/bin/bash
MODELS=${MODELS:-'small_model small_model-cpu large_model large_model-cpu'}
BATCHES=${BATCHES:-'1 16 128 1024'}
MAX_LATENCY=${MAX_LATENCY:-5}

repo_root="$(git rev-parse --show-toplevel)" || repo_root="$PWD"
if [ -z $OUTPUT ]
then
  OUTPUT="$repo_root/qa/benchmark_output"
fi

if [ -z $SHARED_MEM ]
then
  SHARED_MEM="none"
fi

run_benchmark() {
  model="$1"
  batch="$2"
  output_dir="$OUTPUT/$model"
  if [ ! -d "$output_dir" ]
  then
    mkdir -p "$output_dir"
  fi

  output_file="$output_dir/$batch.csv"
  perf_analyzer \
    -a \
    -i GRPC \
    --shared-memory $SHARED_MEM \
    --percentile 99 \
    --binary-search \
    --concurrency-range 1:64:2 \
    -l "$MAX_LATENCY" \
    -m "$model" \
    -b "$batch" \
    -f "$output_file"
}

for model in $MODELS
do
  for batch in $BATCHES
  do
    run_benchmark "$model" "$batch"
  done
done

python3 $repo_root/qa/collate_benchmarks.py $OUTPUT
