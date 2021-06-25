#!/bin/bash
concurrencies='1 4 16 64'
batches='1 16 128 1024 8192'
model="$1"

tp_list='Throughput (infer/s)\n,Batch Size\nConcurrency,'
p99_list='p99 Latency (microseconds)\n,Batch Size\nConcurrency,'
avg_list='Avg Latency (microseconds)\n,Batch Size\nConcurrency,'
for size in $batches
do
  tp_list="${tp_list}${size},"
  p99_list="${p99_list}${size},"
  avg_list="${avg_list}${size},"
done

for conc in $concurrencies
do
  tp_list="${tp_list}\n${conc},"
  p99_list="${p99_list}\n${conc},"
  avg_list="${avg_list}\n${conc},"

  for size in $batches
  do
    output=$(perf_analyzer -i GRPC -m $model -b $size --concurrency-range $conc:$conc)
    throughput=$(echo "$output" | grep Throughput | awk '{ print $2 }')
    p99_latency=$(echo "$output" | grep p99 | awk '{ print $3 }')
    avg_latency=$(echo "$output" | grep Avg\ latency | awk '{ print $3 }')
    tp_list="${tp_list}${throughput},"
    p99_list="${p99_list}${p99_latency},"
    avg_list="${avg_list}${avg_latency},"
  done
done

printf "$tp_list\n\n"
printf "$p99_list\n\n"
printf "$avg_list\n\n"
